"""GPU-free tests for STAGING_REQ admission on the free-list allocator.

When no extent fits, the request is queued on the decode side and answered
with STAGING_RSP (round 0, immediately writable) once a release frees space.
No watermark message is involved anywhere.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingInvariantCounters,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    StagingRoomLifecycle,
    handle_staging_req,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

KV_ARGS = SimpleNamespace(
    page_size=1,
    kv_item_lens=[16, 16],
    total_kv_head_num=1,
    kv_head_num=1,
    engine_rank=0,
)


def _allocator(size):
    allocator = object.__new__(StagingAllocator)
    allocator.total_size = size
    allocator.base_ptr = 0x1000
    allocator.free_extents = [(0, size)]
    allocator.allocations = {}
    allocator.quarantined_allocations = set()
    allocator.quarantine_count = 0
    allocator.invariant_counters = StagingInvariantCounters()
    allocator._quarantine_callback = None
    allocator.next_alloc_id = 0
    allocator.lock = threading.Lock()
    return allocator


def _receiver():
    sock = MagicMock()
    receiver = SimpleNamespace(
        chunk_staging_infos=[],
        _connect_to_bootstrap_server=MagicMock(return_value=(sock, threading.Lock())),
        sock=sock,
    )
    return receiver


def _handler(allocator, rooms):
    handler = object.__new__(DecodeStagingHandler)
    handler.staging_allocator = allocator
    handler.invariant_counters = StagingInvariantCounters()
    handler.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(chunked_prefill_size=8)
    )
    handler._pending_allocs = __import__("collections").deque()
    handler._pending_lock = threading.Lock()
    handler._room_lifecycles = {}
    handler._room_to_receiver = {}
    handler._room_to_decode_req = {}
    room_bootstrap = {}
    for room in rooms:
        handler._room_lifecycles[room] = StagingRoomLifecycle()
        handler._room_to_receiver[room] = _receiver()
        handler._room_to_decode_req[room] = SimpleNamespace(_chunk_events=[])
        room_bootstrap[room] = [{"rank_ip": "10.0.0.1", "rank_port": 1000 + room}]
    handler.kv_manager = SimpleNamespace(
        _staging_ctx=SimpleNamespace(room_bootstrap=room_bootstrap)
    )
    handler.fail_staging_room = MagicMock()
    handler._drain_scatter_bounded = MagicMock()
    return handler, room_bootstrap


def _req(handler, room_bootstrap, room, chunk_idx=0, pages=4, session="s"):
    receiver = handler._room_to_receiver[room]
    handle_staging_req(
        [
            b"STAGING_REQ",
            str(room).encode(),
            str(chunk_idx).encode(),
            str(pages).encode(),
            session.encode(),
        ],
        handler.staging_allocator,
        KV_ARGS,
        1,
        1,
        None,
        handler._room_to_receiver,
        room_bootstrap,
        handler,
        8,
    )
    return receiver


def _rsp_messages(receiver):
    return [c.args[0] for c in receiver.sock.send_multipart.call_args_list]


def _chunk_bytes():
    """Bytes one 4-page chunk needs under KV_ARGS (learned from a big pool)."""
    handler, room_bootstrap = _handler(_allocator(1 << 20), [1])
    _req(handler, room_bootstrap, 1)
    return next(iter(handler.staging_allocator.allocations.values()))[1]


class TestStagingPendingAlloc(CustomTestCase):
    def test_grant_is_immediate_and_round_zero(self):
        handler, rb = _handler(_allocator(_chunk_bytes()), [1])
        receiver = _req(handler, rb, 1)
        info = receiver.chunk_staging_infos[0]
        self.assertGreaterEqual(info[0], 0)
        self.assertEqual(info[2], 0)  # round 0: writable without any watermark
        self.assertEqual(handler._room_lifecycles[1].chunk_states[0], "WRITABLE")
        msgs = _rsp_messages(receiver)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0][0], b"STAGING_RSP")
        self.assertEqual(msgs[0][4], b"0")

    def test_full_pool_queues_and_release_grants_fifo(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1, 2, 3])
        r1 = _req(handler, rb, 1)
        r2 = _req(handler, rb, 2)
        r3 = _req(handler, rb, 3)
        self.assertEqual(handler.pending_alloc_count(), 2)
        for r, room in ((r2, 2), (r3, 3)):
            self.assertEqual(
                handler._room_lifecycles[room].chunk_states[0], "PENDING_ALLOC"
            )
            self.assertEqual(r.chunk_staging_infos[0][0], -1)
            self.assertEqual(_rsp_messages(r), [])
        # Room 1 finishes: its extent goes to room 2 (FIFO), room 3 keeps waiting.
        alloc_id = r1.chunk_staging_infos[0][0]
        handler.release_allocation(alloc_id, handler._room_to_decode_req[1])
        self.assertEqual(handler._room_lifecycles[2].chunk_states[0], "WRITABLE")
        self.assertEqual(r2.chunk_staging_infos[0][2], 0)
        self.assertEqual(len(_rsp_messages(r2)), 1)
        self.assertEqual(_rsp_messages(r2)[0][4], b"0")
        self.assertEqual(handler._room_lifecycles[3].chunk_states[0], "PENDING_ALLOC")
        self.assertEqual(handler.pending_alloc_count(), 1)
        handler.release_allocation(
            r2.chunk_staging_infos[0][0], handler._room_to_decode_req[2]
        )
        self.assertEqual(handler._room_lifecycles[3].chunk_states[0], "WRITABLE")
        self.assertEqual(handler.pending_alloc_count(), 0)

    def test_duplicate_rank_request_while_pending_is_deduplicated(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1, 2])
        _req(handler, rb, 1)
        r2 = _req(handler, rb, 2, session="rank-a")
        _req(handler, rb, 2, session="rank-b")
        self.assertEqual(handler.pending_alloc_count(), 1)
        self.assertEqual(_rsp_messages(r2), [])

    def test_head_of_line_large_request_is_not_starved(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(2 * size), [1, 2, 3, 4])
        r1 = _req(handler, rb, 1)
        r2 = _req(handler, rb, 2)
        _req(handler, rb, 3, pages=8)  # needs 2 * size, queued first
        r4 = _req(handler, rb, 4)  # needs size, queued behind it
        self.assertEqual(handler.pending_alloc_count(), 2)
        handler.release_allocation(
            r1.chunk_staging_infos[0][0], handler._room_to_decode_req[1]
        )
        # One chunk freed: room 3 still does not fit, and room 4 must not jump it.
        self.assertEqual(handler._room_lifecycles[3].chunk_states[0], "PENDING_ALLOC")
        self.assertEqual(handler._room_lifecycles[4].chunk_states[0], "PENDING_ALLOC")
        handler.release_allocation(
            r2.chunk_staging_infos[0][0], handler._room_to_decode_req[2]
        )
        self.assertEqual(handler._room_lifecycles[3].chunk_states[0], "WRITABLE")
        self.assertEqual(handler._room_lifecycles[4].chunk_states[0], "PENDING_ALLOC")
        self.assertEqual(_rsp_messages(r4), [])

    def test_terminal_room_drops_its_pending_request(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1, 2])
        r1 = _req(handler, rb, 1)
        r2 = _req(handler, rb, 2)
        handler._terminalize_room(2, "staging-stall")
        self.assertEqual(handler.pending_alloc_count(), 0)
        handler.release_allocation(
            r1.chunk_staging_infos[0][0], handler._room_to_decode_req[1]
        )
        self.assertEqual(_rsp_messages(r2), [])
        self.assertEqual(handler.staging_allocator.free_bytes(), size)

    def test_oversized_chunk_is_answered_immediately(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1])
        receiver = _req(handler, rb, 1, pages=8)  # needs 2 * size > pool
        self.assertEqual(handler._room_lifecycles[1].chunk_states[0], "OVERSIZED")
        self.assertEqual(
            receiver.chunk_staging_infos[0][1], StagingAllocator.ALLOC_OVERSIZED
        )
        msgs = _rsp_messages(receiver)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0][3], str(StagingAllocator.ALLOC_OVERSIZED).encode())
        self.assertEqual(handler.pending_alloc_count(), 0)


    def test_release_under_room_lock_services_same_room_head(self):
        # Regression: freeing under a room's lifecycle lock used to service the
        # queue inline; with that room at the queue head the non-reentrant lock
        # deadlocked. Servicing now happens after the lock is dropped.
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1])
        r1 = _req(handler, rb, 1, chunk_idx=0)
        _req(handler, rb, 1, chunk_idx=1)
        self.assertEqual(handler._room_lifecycles[1].chunk_states[1], "PENDING_ALLOC")
        done = threading.Event()

        def release():
            handler.release_unwritten_chunks(1)
            done.set()

        worker = threading.Thread(target=release, daemon=True)
        worker.start()
        self.assertTrue(done.wait(5.0), "release_unwritten_chunks deadlocked")
        self.assertEqual(handler._room_lifecycles[1].chunk_states[1], "WRITABLE")
        self.assertEqual(handler.pending_alloc_count(), 0)
        self.assertEqual(len(_rsp_messages(r1)), 2)

    def test_unattainable_queued_request_fails_room_instead_of_blocking(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(2 * size), [1, 2, 3])
        _req(handler, rb, 1)
        _req(handler, rb, 2)
        # Room 3 needs two chunks' worth; attainable now, so it queues.
        _req(handler, rb, 3, pages=8)
        self.assertEqual(handler.pending_alloc_count(), 1)
        alloc1 = handler._room_to_receiver[1].chunk_staging_infos[0][0]
        alloc2 = handler._room_to_receiver[2].chunk_staging_infos[0][0]
        self.assertTrue(handler.staging_allocator.quarantine(alloc1, "test"))
        # A release services the queue: room 3 can never fit any more.
        handler.release_allocation(alloc2, handler._room_to_decode_req[2])
        self.assertEqual(handler.pending_alloc_count(), 0)
        handler.fail_staging_room.assert_called_once()
        self.assertEqual(handler.fail_staging_room.call_args.args[0], 3)
        self.assertIn(
            "STAGING_ALLOC_UNATTAINABLE", handler.fail_staging_room.call_args.args[1]
        )

    def test_oversized_uses_attainable_extent_not_pool_total(self):
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(2 * size), [1, 2])
        _req(handler, rb, 1)
        alloc1 = handler._room_to_receiver[1].chunk_staging_infos[0][0]
        self.assertTrue(handler.staging_allocator.quarantine(alloc1, "test"))
        self.assertEqual(handler.staging_allocator.max_attainable_extent(), size)
        r2 = _req(handler, rb, 2, pages=8)
        self.assertEqual(handler._room_lifecycles[2].chunk_states[0], "OVERSIZED")
        self.assertEqual(handler.pending_alloc_count(), 0)
        msgs = _rsp_messages(r2)
        self.assertEqual(len(msgs), 1)

    def test_enqueue_race_is_closed_by_servicing_after_enqueue(self):
        # The pool looks full when the request is checked, but a release lands
        # before the request is enqueued. Servicing right after the enqueue must
        # grant it instead of leaving it queued until the next release.
        size = _chunk_bytes()
        handler, rb = _handler(_allocator(size), [1])
        allocator = handler.staging_allocator
        real_assign = allocator.assign
        calls = {"n": 0}

        def assign_full_once(required):
            calls["n"] += 1
            if calls["n"] == 1:
                return None
            return real_assign(required)

        allocator.assign = assign_full_once
        r1 = _req(handler, rb, 1)
        self.assertEqual(handler._room_lifecycles[1].chunk_states[0], "WRITABLE")
        self.assertEqual(handler.pending_alloc_count(), 0)
        self.assertEqual(len(_rsp_messages(r1)), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
