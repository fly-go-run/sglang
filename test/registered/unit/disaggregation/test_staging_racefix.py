"""GPU-free tests for staging race-safety invariants."""

import threading
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingInvariantCounters,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    PrefillStagingContext,
    StagingRoomLifecycle,
    handle_staging_req,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVSender,
    TransferInfo,
    _StagingSenderRoomState,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cpu_allocator(size=128):
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
    allocator._scatter_stream = None
    return allocator


class _FakeEvent:
    def record(self, stream):
        self.stream = stream


class TestStagingRaceFix(CustomTestCase):
    @staticmethod
    def _make_coverage_handler(expected_pages=899):
        handler = object.__new__(DecodeStagingHandler)
        handler.decode_tp = 1
        handler.tp_rank = 0
        handler._requires_last_writer_slots = True
        handler.invariant_counters = StagingInvariantCounters()
        handler.staging_allocator = SimpleNamespace(_scatter_stream=object())
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(attn_tp_size=1),
            chunk_staging_infos=[(17, 256, 0, 512, expected_pages)],
        )
        decode_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=19),
            kv_receiver=receiver,
            _staging_progress_lock=threading.Lock(),
            _staging_stall_failed=False,
            _staging_stall_since=None,
            _staging_data_started=True,
            _staging_scatter_done=False,
            _staging_all_success=False,
            _staging_success_ts=0.0,
            _chunk_events=[],
        )
        lifecycle = StagingRoomLifecycle(
            chunk_states={0: "WRITABLE"},
            chunk_geometry={0: (0, expected_pages)},
        )
        handler._room_to_decode_req = {19: decode_req}
        handler._room_to_receiver = {19: receiver}
        handler._room_lifecycles = {19: lifecycle}
        handler._scatter_region = MagicMock(return_value=True)
        handler._free_allocation = MagicMock()
        handler.fail_staging_room = MagicMock()
        counts = defaultdict(lambda: defaultdict(set))
        return handler, lifecycle, counts

    def test_quarantined_allocation_never_frees_or_advances_watermark(self):
        allocator = _cpu_allocator()
        first = allocator.assign(64)[0]
        second = allocator.assign(64)[0]
        callback = MagicMock()
        allocator.set_quarantine_callback(callback)

        self.assertTrue(allocator.quarantine(first, "test"))
        self.assertTrue(allocator.free(second))
        watermark = allocator.get_watermark()
        self.assertFalse(allocator.free(first))

        self.assertIn(first, allocator.allocations)
        self.assertEqual(allocator.get_watermark(), watermark)
        self.assertTrue(allocator.quarantine(first, "duplicate"))
        self.assertEqual(allocator.quarantine_count, 1)
        self.assertEqual(allocator.invariant_counters.get("free_quarantined"), 1)
        callback.assert_called_once_with(first, "test")

    def test_sender_skips_done_destination_without_second_post(self):
        mgr = object.__new__(NixlKVManager)
        mgr._staging_ctx = PrefillStagingContext(
            invariant_counters=StagingInvariantCounters()
        )
        mgr.kv_args = SimpleNamespace(engine_rank=0)
        mgr.send_kvcache_staged = MagicMock(return_value="handle")
        strategy = SimpleNamespace(
            check_ready=MagicMock(return_value=(True, 0, 32, 0, 96)),
            staging_buffer=object(),
            full_chunk_pages=2,
        )
        chunk = TransferKVChunk(
            room=7,
            prefill_kv_indices=np.array([1, 2], dtype=np.int32),
            index_slice=slice(0, 2),
            is_last_chunk=False,
            chunk_id=3,
            prefill_aux_index=None,
            state_indices=None,
        )
        req = SimpleNamespace(room=7, agent_name="decode")
        dst = SimpleNamespace(
            staging=SimpleNamespace(base_ptr=0x2000, total_size=4096),
            gpu_id=0,
            decode_tp_rank=0,
            decode_tp_size=1,
            dst_kv_item_len=128,
        )

        first = mgr._do_staging_transfer(
            strategy, chunk, chunk.prefill_kv_indices, req, dst, MagicMock()
        )
        mgr._mark_staging_send_done(7, 3, "decode")
        second = mgr._do_staging_transfer(
            strategy, chunk, chunk.prefill_kv_indices, req, dst, MagicMock()
        )

        self.assertEqual(first, ("handle", False, True))
        self.assertEqual(second, (None, False, True))
        mgr.send_kvcache_staged.assert_called_once()
        self.assertEqual(mgr._staging_ctx.invariant_counters.get("duplicate_post"), 0)

    def test_deferred_early_chunk_fences_last_chunk_room_completion(self):
        room = 41
        mgr = object.__new__(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.enable_staging = True
        mgr.kv_buffer_tensors = object()
        mgr.attn_tp_size = 2
        mgr.is_mla_backend = False
        mgr.kv_args = SimpleNamespace(engine_rank=0)
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.req_to_decode_prefix_len = {room: 0}
        mgr._sender_room_lock = threading.RLock()
        mgr._staging_sender_rooms = {}
        mgr._staging_ctx = PrefillStagingContext(
            invariant_counters=StagingInvariantCounters()
        )
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}

        def make_req(agent_name):
            return TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=9000,
                agent_name=agent_name,
                dst_kv_indices=np.array([10, 11], dtype=np.int32),
                dst_aux_index=0,
                required_dst_info_num=2,
                dst_state_indices=[],
            )

        mgr.transfer_infos = {
            room: {agent: make_req(agent) for agent in ("decode-a", "decode-b")}
        }
        mgr.decode_kv_args_table = {
            agent: SimpleNamespace(
                decode_tp_size=1,
                staging=SimpleNamespace(base_ptr=0x8000, total_size=4096),
                dst_aux_ptrs=[0],
            )
            for agent in ("decode-a", "decode-b")
        }
        mgr._prefetch_staging_reqs = MagicMock()
        mgr._try_create_staging_strategy = MagicMock(return_value=object())
        mgr._wait_for_xfers = MagicMock()
        mgr.send_aux = MagicMock(return_value="aux-handle")

        class RaceQueue:
            def __init__(self):
                self.items = deque()
                self.get_count = 0

            def put(self, item):
                self.items.append(item)

            def get(self):
                if self.get_count == 2:
                    state = mgr._staging_sender_rooms[room]
                    self_case.assertEqual(state.registered_chunks, {0, 1})
                    self_case.assertEqual(state.completed_chunks, {1})
                    self_case.assertEqual(state.last_chunk_id, 1)
                    self_case.assertNotEqual(mgr.request_status[room], KVPoll.Success)
                    self_case.assertIn(room, mgr.transfer_infos)
                self.get_count += 1
                if not self.items:
                    raise SystemExit()
                return self.items.popleft()

        self_case = self
        queue = RaceQueue()
        mgr.transfer_queues = [queue]
        attempts = defaultdict(int)

        def do_staging(_strategy, chunk, _indices, req, _dst, worker_queue):
            key = (chunk.chunk_id, req.agent_name)
            attempts[key] += 1
            if key == (0, "decode-a") and attempts[key] == 1:
                worker_queue.put(chunk)
                return (None, True, False)
            return (None, False, True)

        mgr._do_staging_transfer = MagicMock(side_effect=do_staging)
        mgr.add_transfer_request(
            room,
            np.array([1], dtype=np.int32),
            slice(0, 1),
            False,
            0,
        )
        mgr.add_transfer_request(
            room,
            np.array([2], dtype=np.int32),
            slice(1, 2),
            True,
            1,
            aux_index=0,
        )

        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue, staging_buffer=object(), worker_idx=0)

        self.assertEqual(mgr.request_status[room], KVPoll.Success)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, mgr._staging_sender_rooms)
        self.assertEqual(attempts[(0, "decode-a")], 2)
        self.assertEqual(attempts[(0, "decode-b")], 1)
        self.assertEqual(attempts[(1, "decode-a")], 1)
        self.assertEqual(attempts[(1, "decode-b")], 1)

    def test_staging_sender_duplicate_completion_and_abort_are_terminal(self):
        room = 43
        mgr = object.__new__(NixlKVManager)
        mgr.enable_staging = True
        mgr._sender_room_lock = threading.RLock()
        mgr._staging_sender_rooms = {
            room: _StagingSenderRoomState(registered_chunks={0, 1}, last_chunk_id=1)
        }
        mgr.request_status = {room: KVPoll.Transferring}
        mgr.transfer_infos = {room: {"decode": object()}}
        mgr.req_to_decode_prefix_len = {room: 0}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}

        self.assertFalse(mgr._finish_staging_sender_chunk(room, 1))
        self.assertFalse(mgr._finish_staging_sender_chunk(room, 1))
        self.assertEqual(mgr._staging_sender_rooms[room].completed_chunks, {1})

        self.assertTrue(mgr._handle_abort_notification([b"ABORT", b"43"]))

        self.assertEqual(mgr.request_status[room], KVPoll.Failed)
        self.assertNotIn(room, mgr._staging_sender_rooms)
        self.assertFalse(mgr._finish_staging_sender_chunk(room, 0))
        self.assertEqual(mgr.request_status[room], KVPoll.Failed)

    def test_clear_then_late_completion_does_not_revive_room(self):
        room = 44
        mgr = object.__new__(NixlKVManager)
        mgr.enable_staging = True
        mgr._sender_room_lock = threading.RLock()
        mgr._staging_sender_rooms = {
            room: _StagingSenderRoomState(
                registered_chunks={0, 1}, completed_chunks={1}, last_chunk_id=1
            )
        }
        mgr.request_status = {room: KVPoll.Transferring}
        mgr.transfer_infos = {room: {"decode": object()}}
        mgr.req_to_decode_prefix_len = {room: 0}
        mgr._staging_ctx = PrefillStagingContext(
            invariant_counters=StagingInvariantCounters()
        )
        sender = object.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.kv_mgr = mgr

        sender.clear()
        self.assertFalse(mgr._finish_staging_sender_chunk(room, 0))

        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr._staging_sender_rooms)

    def test_writer_slot_dedup_and_scatter_once_claim(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.decode_tp = 1
        handler.kv_manager = SimpleNamespace()
        handler._requires_last_writer_slots = True
        handler.tp_rank = 0
        handler.invariant_counters = StagingInvariantCounters()
        handler.staging_allocator = SimpleNamespace(_scatter_stream=object())
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(attn_tp_size=2),
            chunk_staging_infos=[(11, 256, 0, 512, 4)],
        )
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            _staging_progress_lock=threading.Lock(),
            _staging_stall_failed=False,
            _staging_stall_since=None,
            _chunk_events=[],
        )
        lifecycle = StagingRoomLifecycle(
            chunk_states={0: "WRITABLE"},
            chunk_geometry={0: (0, 4)},
        )
        handler._room_to_decode_req = {9: decode_req}
        handler._room_to_receiver = {9: receiver}
        handler._room_lifecycles = {9: lifecycle}
        handler._scatter_region = MagicMock(return_value=True)
        counts = defaultdict(lambda: defaultdict(set))

        self.assertEqual(
            handler.handle_chunk_arrived(9, 0, 0, 4, 0, "peer-a", counts),
            (True, False),
        )
        self.assertEqual(
            handler.handle_chunk_arrived(9, 0, 0, 4, 2, "peer-b", counts),
            (False, False),
        )
        self.assertEqual(
            handler.handle_chunk_arrived(9, 0, 0, 4, 1, "peer-c", counts),
            (True, True),
        )
        with patch(
            "sglang.srt.disaggregation.common.staging_handler.torch.cuda.Event",
            _FakeEvent,
        ):
            self.assertTrue(handler.submit_chunk_scatter(9, 0, 0, 4))
            self.assertTrue(handler.submit_chunk_scatter(9, 0, 0, 4))
        self.assertEqual(
            handler.handle_chunk_arrived(9, 0, 0, 4, 0, "peer-a", counts),
            (False, False),
        )

        handler._scatter_region.assert_called_once()
        self.assertEqual(lifecycle.chunk_states[0], "SCATTER_SUBMITTED")

    def test_partial_intervals_cover_full_chunk_before_single_scatter(self):
        handler, lifecycle, counts = self._make_coverage_handler()

        self.assertEqual(
            handler.handle_chunk_arrived(19, 0, 0, 567, 0, "peer", counts),
            (True, False),
        )
        self.assertTrue(handler._room_to_decode_req[19]._staging_data_started)
        self.assertEqual(
            handler.handle_chunk_arrived(19, 0, 567, 332, 0, "peer", counts),
            (True, True),
        )
        with patch(
            "sglang.srt.disaggregation.common.staging_handler.torch.cuda.Event",
            _FakeEvent,
        ):
            self.assertTrue(handler.submit_chunk_scatter(19, 0, 0, 899))
            self.assertTrue(handler.submit_chunk_scatter(19, 0, 0, 899))

        handler._scatter_region.assert_called_once_with(256, 0, 899, ANY)
        self.assertEqual(lifecycle.chunk_states[0], "SCATTER_SUBMITTED")

    def test_out_of_bounds_interval_fails_room(self):
        handler, lifecycle, counts = self._make_coverage_handler()

        self.assertEqual(
            handler.handle_chunk_arrived(19, 0, 0, 900, 0, "peer", counts),
            (False, False),
        )

        self.assertTrue(lifecycle.terminal)
        handler.fail_staging_room.assert_called_once()
        handler._scatter_region.assert_not_called()

    def test_coverage_hole_does_not_scatter_and_completion_fails_room(self):
        handler, lifecycle, counts = self._make_coverage_handler()

        self.assertEqual(
            handler.handle_chunk_arrived(19, 0, 0, 567, 0, "peer", counts),
            (True, False),
        )
        self.assertEqual(
            handler.handle_chunk_arrived(19, 0, 568, 331, 0, "peer", counts),
            (True, False),
        )
        # All-ranks Success only records the transport-level fact; the hole
        # keeps the allocation live, so the room stays open (the stall
        # watchdog owns it from here) and nothing is scattered.
        decode_req = handler._room_to_decode_req[19]
        self.assertTrue(handler.submit_last_scatter_async(19))
        self.assertTrue(decode_req._staging_all_success)
        handler.advance_scatter(decode_req)
        self.assertFalse(decode_req._staging_scatter_done)
        self.assertFalse(handler.is_done(decode_req))
        self.assertFalse(lifecycle.terminal)
        handler.fail_staging_room.assert_not_called()
        handler._scatter_region.assert_not_called()
        handler._free_allocation.assert_not_called()

    def test_terminal_last_scatter_does_not_dereference_cleared_receiver(self):
        handler, lifecycle, _counts = self._make_coverage_handler()
        lifecycle.terminal = True
        handler._room_to_decode_req[19].kv_receiver = None

        self.assertFalse(handler.submit_last_scatter_async(19))

    def test_legacy_shared_arrival_api_still_submits_scatter(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.decode_tp = 1
        handler.kv_manager = SimpleNamespace()
        handler._requires_last_writer_slots = False
        receiver = SimpleNamespace(
            prefill_info=SimpleNamespace(attn_tp_size=1),
            chunk_staging_infos=[(3, 64, 0, 128, 2)],
        )
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            _staging_progress_lock=threading.Lock(),
            _staging_stall_failed=False,
            _staging_stall_since=None,
            _staging_all_success=False,
            _staging_success_ts=0.0,
        )
        handler._room_to_decode_req = {4: decode_req}
        handler._room_lifecycles = {
            4: StagingRoomLifecycle(
                chunk_states={0: "WRITABLE"},
                chunk_geometry={0: (0, 2)},
            )
        }
        handler.submit_chunk_scatter = MagicMock(return_value=True)
        counts = defaultdict(lambda: defaultdict(list))

        self.assertEqual(
            handler.handle_chunk_arrived(4, 0, 0, 2, "session-a", counts),
            (True, True),
        )
        handler.submit_chunk_scatter.assert_called_once_with(4, 0, 0, 2)
        handler.submit_chunk_scatter.reset_mock()
        # Mooncake still calls this on all-ranks Success; it only records the
        # fact and never submits a second scatter for the same chunk.
        self.assertTrue(handler.submit_last_scatter_async(4))
        self.assertTrue(decode_req._staging_all_success)
        handler.submit_chunk_scatter.assert_not_called()

    def test_late_staging_req_after_scatter_is_rejected(self):
        allocator = _cpu_allocator()
        allocator.assign = MagicMock()
        handler = SimpleNamespace(
            _room_lifecycles={
                5: StagingRoomLifecycle(
                    chunk_states={0: "SCATTER_DONE"},
                    chunk_geometry={0: (0, 4)},
                )
            },
            invariant_counters=StagingInvariantCounters(),
        )
        receiver = SimpleNamespace(chunk_staging_infos=[(-1, -1, 0, -1, 0)])
        kv_args = SimpleNamespace(
            page_size=1,
            kv_item_lens=[16, 16],
            total_kv_head_num=1,
            kv_head_num=1,
            engine_rank=0,
        )

        handle_staging_req(
            [b"STAGING_REQ", b"5", b"0", b"4", b"session"],
            allocator,
            kv_args,
            1,
            1,
            None,
            {5: receiver},
            {},
            handler,
            4,
        )

        allocator.assign.assert_not_called()
        self.assertEqual(handler.invariant_counters.get("alloc_after_terminal"), 1)

    @staticmethod
    def _make_quarantine_handler(total_size=1000, quarantined_bytes=0):
        handler = object.__new__(DecodeStagingHandler)
        handler.scheduler = SimpleNamespace(set_system_status=MagicMock())
        handler._quarantine_exit_lock = threading.Lock()
        handler._quarantine_exit_timer = None
        handler._fatal_shutdown = MagicMock()
        handler.staging_allocator = SimpleNamespace(
            quarantine_count=1,
            total_size=total_size,
            quarantined_bytes=MagicMock(return_value=quarantined_bytes),
        )
        timers = []

        class FakeTimer:
            def __init__(self, delay, callback):
                self.delay = delay
                self.callback = callback
                self.daemon = False
                timers.append(self)

            def start(self):
                pass

        return handler, timers, FakeTimer

    def test_invariant_violation_quarantine_arms_one_process_exit_timer(self):
        handler, timers, FakeTimer = self._make_quarantine_handler()
        with (
            patch(
                "sglang.srt.disaggregation.common.staging_handler.random.uniform",
                return_value=12.5,
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_handler.threading.Timer",
                FakeTimer,
            ),
        ):
            handler._on_quarantine(4, "[STAGING_GEOMETRY] overlapping writer")
            handler._on_quarantine(5, "[STAGING_WRITER] invalid slot")

        handler.scheduler.set_system_status.assert_called_once_with("notready")
        self.assertEqual(len(timers), 1)
        self.assertEqual(timers[0].delay, 12.5)
        timers[0].callback()
        handler._fatal_shutdown.assert_called_once_with(
            -1, 1, reason="staging-quarantine"
        )

    def test_stall_quarantine_below_capacity_threshold_only_isolates(self):
        handler, timers, FakeTimer = self._make_quarantine_handler(
            total_size=1000, quarantined_bytes=50
        )
        with patch(
            "sglang.srt.disaggregation.common.staging_handler.threading.Timer",
            FakeTimer,
        ):
            handler._on_quarantine(4, "staging-stall")
            handler._on_quarantine(5, "decode-transfer-failed")
            handler._on_quarantine(6, "room-release")

        self.assertEqual(timers, [])
        handler.scheduler.set_system_status.assert_not_called()
        handler._fatal_shutdown.assert_not_called()

    def test_quarantined_capacity_over_threshold_arms_process_exit(self):
        handler, timers, FakeTimer = self._make_quarantine_handler(
            total_size=1000, quarantined_bytes=150
        )
        with (
            patch(
                "sglang.srt.disaggregation.common.staging_handler.random.uniform",
                return_value=3.0,
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_handler.threading.Timer",
                FakeTimer,
            ),
        ):
            handler._on_quarantine(4, "staging-stall")

        handler.scheduler.set_system_status.assert_called_once_with("notready")
        self.assertEqual(len(timers), 1)
        timers[0].callback()
        handler._fatal_shutdown.assert_called_once_with(
            -1, 1, reason="staging-quarantine"
        )

    def test_allocator_reports_quarantined_bytes_and_fires_per_allocation(self):
        allocator = _cpu_allocator(size=256)
        first = allocator.assign(64)[0]
        second = allocator.assign(32)[0]
        callback = MagicMock()
        allocator.set_quarantine_callback(callback)

        self.assertTrue(allocator.quarantine(first, "staging-stall"))
        self.assertTrue(allocator.quarantine(second, "room-release"))
        self.assertTrue(allocator.quarantine(second, "duplicate"))
        self.assertEqual(allocator.quarantine_count, 2)
        self.assertEqual(allocator.quarantined_bytes(), 96)
        self.assertEqual(callback.call_count, 2)

    def test_scatter_drain_timeout_requests_process_exit(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.staging_allocator = SimpleNamespace(
            _scatter_stream=SimpleNamespace(query=MagicMock(return_value=False))
        )
        handler._fatal_shutdown = MagicMock()

        with patch(
            "sglang.srt.disaggregation.common.staging_handler."
            "STAGING_SCATTER_DRAIN_TIMEOUT_S",
            0,
        ):
            with self.assertRaisesRegex(RuntimeError, "STAGING_SCATTER_DRAIN_TIMEOUT"):
                handler._drain_scatter_bounded(12)

        handler._fatal_shutdown.assert_called_once_with(
            12, 1, reason="staging-scatter-drain-timeout"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
