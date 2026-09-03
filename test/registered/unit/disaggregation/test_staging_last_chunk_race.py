"""GPU-free tests for arrival-driven staging completion.

Chunked prefill can split a short prompt at the batch token budget, so the
request's final logical chunk shares one staging chunk with the chunk before
it. Notifications arrive in transfer-completion order, not post order, so the
staging layer must never decide "this room is complete" from which
notification happened to land last. Completion is resource-driven instead:
all ranks reported Success, every scatter event fired, no allocation is live.
"""

import itertools
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.staging_buffer import StagingInvariantCounters
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    StagingRoomLifecycle,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, TransferStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 9
EVENT_PATCH = "sglang.srt.disaggregation.common.staging_handler.torch.cuda.Event"


class _FakeEvent:
    def __init__(self, done=True):
        self.done = done

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.done


def _make_handler(geometry, chunk_infos, num_writers=1):
    handler = object.__new__(DecodeStagingHandler)
    handler.decode_tp = 1
    handler.tp_rank = 0
    handler.kv_manager = MagicMock()
    handler.kv_manager.transfer_statuses = {}
    handler.invariant_counters = StagingInvariantCounters()
    handler.staging_allocator = SimpleNamespace(
        _scatter_stream=object(), allocations={}
    )
    receiver = SimpleNamespace(
        prefill_info=SimpleNamespace(attn_tp_size=num_writers),
        require_staging=True,
        chunk_staging_infos=list(chunk_infos),
        conclude_state=None,
    )
    decode_req = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=ROOM, rid="rid-9"),
        kv_receiver=receiver,
        _staging_progress_lock=threading.Lock(),
        _staging_stall_failed=False,
        _staging_stall_since=None,
        _staging_data_started=False,
        _staging_scatter_done=False,
        _staging_all_success=False,
        _staging_success_ts=0.0,
        _chunk_events=[],
    )
    lifecycle = StagingRoomLifecycle(
        chunk_states={idx: "WRITABLE" for idx in geometry},
        chunk_geometry=dict(geometry),
    )
    handler._room_to_decode_req = {ROOM: decode_req}
    handler._room_to_receiver = {ROOM: receiver}
    handler._room_lifecycles = {ROOM: lifecycle}
    handler._scatter_region = MagicMock(return_value=True)
    handler._free_allocation = MagicMock()
    handler.fail_staging_room = MagicMock()
    return handler, decode_req, receiver, lifecycle


class TestStagingArrivalDrivenCompletion(CustomTestCase):
    def _run_arrivals(self, handler, arrivals):
        """Feed (page_start, num_pages, engine_rank) arrivals; scatter on coverage."""
        counts = defaultdict(lambda: defaultdict(set))
        submitted = 0
        for page_start, num_pages, engine_rank in arrivals:
            accepted, complete = handler.handle_chunk_arrived(
                ROOM,
                0,
                page_start,
                num_pages,
                engine_rank,
                f"peer-{engine_rank}",
                counts,
            )
            self.assertTrue(accepted)
            if complete:
                with patch(EVENT_PATCH, _FakeEvent):
                    self.assertTrue(handler.submit_chunk_scatter(ROOM, 0, 0, 4))
                submitted += 1
        return submitted

    def test_two_logical_chunks_two_writers_every_arrival_order(self):
        """24 arrival orders of {chunk A, chunk B} x {writer 0, writer 1}."""
        arrivals = [(0, 2, 0), (2, 2, 0), (0, 2, 1), (2, 2, 1)]
        for order in itertools.permutations(arrivals):
            with self.subTest(order=order):
                handler, decode_req, receiver, lifecycle = _make_handler(
                    geometry={0: (0, 4)},
                    chunk_infos=[(11, 256, 0, 512, 4)],
                    num_writers=2,
                )
                # Success can be recorded before or after the final arrival;
                # both must converge.
                handler.submit_last_scatter_async(ROOM)
                submitted = self._run_arrivals(handler, order)
                self.assertEqual(submitted, 1)
                self.assertEqual(handler._scatter_region.call_count, 1)
                self.assertEqual(lifecycle.chunk_states[0], "SCATTER_SUBMITTED")
                self.assertFalse(handler.is_done(decode_req))
                handler.advance_scatter(decode_req)
                self.assertEqual(lifecycle.chunk_states[0], "SCATTER_DONE")
                self.assertEqual(receiver.chunk_staging_infos[0], (-1, -1, 0, -1, 0))
                handler._free_allocation.assert_called_once_with(11, decode_req)
                self.assertTrue(decode_req._staging_scatter_done)
                self.assertTrue(handler.is_done(decode_req))
                self.assertFalse(decode_req._staging_stall_failed)
                handler.fail_staging_room.assert_not_called()

    def test_success_recorded_after_scatter_event_completes(self):
        handler, decode_req, _, _ = _make_handler(
            geometry={0: (0, 4)}, chunk_infos=[(11, 256, 0, 512, 4)]
        )
        self.assertEqual(self._run_arrivals(handler, [(0, 4, 0)]), 1)
        handler.advance_scatter(decode_req)
        # Event fired and allocation freed, but the transport has not yet
        # reported all-ranks Success: the room must stay open.
        self.assertFalse(decode_req._staging_scatter_done)
        self.assertTrue(handler.submit_last_scatter_async(ROOM))
        handler.advance_scatter(decode_req)
        self.assertTrue(handler.is_done(decode_req))

    def test_two_staging_chunks_complete_only_when_both_scattered(self):
        handler, decode_req, receiver, lifecycle = _make_handler(
            geometry={0: (0, 4), 1: (4, 2)},
            chunk_infos=[(11, 256, 0, 512, 4), (12, 512, 0, 768, 2)],
        )
        counts = defaultdict(lambda: defaultdict(set))
        handler.submit_last_scatter_async(ROOM)
        self.assertEqual(
            handler.handle_chunk_arrived(ROOM, 1, 4, 2, 0, "peer", counts), (True, True)
        )
        with patch(EVENT_PATCH, _FakeEvent):
            self.assertTrue(handler.submit_chunk_scatter(ROOM, 1, 4, 2))
        handler.advance_scatter(decode_req)
        self.assertEqual(lifecycle.chunk_states[1], "SCATTER_DONE")
        # Chunk 0 still holds its allocation: not done.
        self.assertFalse(handler.is_done(decode_req))
        self.assertEqual(
            handler.handle_chunk_arrived(ROOM, 0, 0, 4, 0, "peer", counts), (True, True)
        )
        with patch(EVENT_PATCH, _FakeEvent):
            self.assertTrue(handler.submit_chunk_scatter(ROOM, 0, 0, 4))
        handler.advance_scatter(decode_req)
        self.assertTrue(handler.is_done(decode_req))
        self.assertEqual(handler._free_allocation.call_count, 2)
        self.assertTrue(all(info[0] < 0 for info in receiver.chunk_staging_infos))

    def test_pending_event_keeps_room_open(self):
        handler, decode_req, _, _ = _make_handler(
            geometry={0: (0, 4)}, chunk_infos=[(11, 256, 0, 512, 4)]
        )
        counts = defaultdict(lambda: defaultdict(set))
        handler.handle_chunk_arrived(ROOM, 0, 0, 4, 0, "peer", counts)
        with patch(EVENT_PATCH, lambda: _FakeEvent(done=False)):
            self.assertTrue(handler.submit_chunk_scatter(ROOM, 0, 0, 4))
        handler.submit_last_scatter_async(ROOM)
        handler.advance_scatter(decode_req)
        self.assertFalse(handler.is_done(decode_req))
        decode_req._chunk_events[0][0].done = True
        handler.advance_scatter(decode_req)
        self.assertTrue(handler.is_done(decode_req))

    def test_partial_coverage_never_scatters_and_stalls(self):
        handler, decode_req, _, lifecycle = _make_handler(
            geometry={0: (0, 4)}, chunk_infos=[(11, 256, 0, 512, 4)], num_writers=2
        )
        counts = defaultdict(lambda: defaultdict(set))
        self.assertEqual(
            handler.handle_chunk_arrived(ROOM, 0, 0, 4, 0, "peer-0", counts),
            (True, False),
        )
        handler.submit_last_scatter_async(ROOM)
        with patch(
            "sglang.srt.disaggregation.common.staging_handler.time.monotonic",
            return_value=0.0,
        ):
            handler.advance_scatter(decode_req)
        self.assertFalse(handler.is_done(decode_req))
        handler._scatter_region.assert_not_called()
        with patch(
            "sglang.srt.disaggregation.common.staging_handler.time.monotonic",
            return_value=1000.0,
        ):
            handler._terminalize_room = MagicMock(return_value=set())
            handler.staging_allocator.get_watermark = MagicMock(return_value=(0, 0))
            handler.advance_scatter(decode_req)
        self.assertTrue(decode_req._staging_stall_failed)

    def test_nokv_releases_unwritten_chunks(self):
        handler, decode_req, receiver, lifecycle = _make_handler(
            geometry={0: (0, 4)}, chunk_infos=[(11, 256, 0, 512, 4)]
        )
        handler.release_unwritten_chunks(ROOM)
        self.assertEqual(receiver.chunk_staging_infos[0], (-1, -1, 0, -1, 0))
        self.assertEqual(lifecycle.chunk_states[0], "SCATTER_DONE")
        handler._free_allocation.assert_called_once_with(11, decode_req)
        handler.submit_last_scatter_async(ROOM)
        handler.advance_scatter(decode_req)
        self.assertTrue(handler.is_done(decode_req))
        # A late writer for a released chunk is rejected, not scattered.
        counts = defaultdict(lambda: defaultdict(set))
        self.assertEqual(
            handler.handle_chunk_arrived(ROOM, 0, 0, 4, 0, "peer", counts),
            (False, False),
        )
        handler.fail_staging_room.assert_called_once()

    def test_nixl_stg_notification_ignores_tag_for_scatter(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_staging = True
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr._chunk_writer_counts = {}
        mgr.required_prefill_response_num_table = {ROOM: 1}
        handler = MagicMock()
        handler.is_staging_room.return_value = True
        handler.get_chunk_geometry.return_value = (0, 4)
        mgr._staging_handler = handler
        mgr._handle_staging_chunk_arrived = MagicMock(return_value=(True, True))

        # Tag says is_last=0: scatter is still submitted on coverage.
        mgr._handle_stg_notification(
            [str(ROOM), "stg", "0", "0", "0", "0", "0", "2", "agent"], ROOM, "peer"
        )
        handler.submit_chunk_scatter.assert_called_once_with(ROOM, 0, 0, 4)
        handler.submit_last_scatter_async.assert_not_called()

        # Tag says is_last=1 and aux already landed: all-ranks Success is
        # recorded after the scatter submission, never before it.
        handler.reset_mock()
        handler.is_staging_room.return_value = True
        handler.get_chunk_geometry.return_value = (0, 4)
        mgr.transfer_statuses[ROOM].received_aux = True
        mgr._handle_stg_notification(
            [str(ROOM), "stg", "1", "1", "0", "0", "2", "2", "agent"], ROOM, "peer"
        )
        handler.submit_chunk_scatter.assert_called_once_with(ROOM, 0, 0, 4)
        handler.submit_last_scatter_async.assert_called_once_with(ROOM)
        self.assertEqual(mgr.transfer_statuses[ROOM].expected_kvs_per_pp[0], 2)

    def test_nixl_aux_nokv_releases_when_every_rank_reports_nokv(self):
        mgr = object.__new__(NixlKVManager)
        mgr.enable_staging = True
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr._chunk_writer_counts = {}
        mgr.required_prefill_response_num_table = {ROOM: 2}
        handler = MagicMock()
        handler.is_staging_room.return_value = True
        mgr._staging_handler = handler

        mgr._handle_aux_notification(ROOM, [str(ROOM), "aux", "nokv", "0"])
        handler.release_unwritten_chunks.assert_not_called()
        mgr._handle_aux_notification(ROOM, [str(ROOM), "aux", "nokv", "1"])
        handler.release_unwritten_chunks.assert_called_once_with(ROOM)
        handler.submit_last_scatter_async.assert_called_once_with(ROOM)


if __name__ == "__main__":
    import unittest

    unittest.main()
