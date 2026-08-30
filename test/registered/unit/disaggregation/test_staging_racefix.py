"""GPU-free tests for staging race-safety invariants."""

import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np

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
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cpu_allocator(size=128):
    allocator = object.__new__(StagingAllocator)
    allocator.total_size = size
    allocator.base_ptr = 0x1000
    allocator.head = 0
    allocator.round = 0
    allocator.allocations = {}
    allocator.alloc_order = []
    allocator.quarantined_allocations = set()
    allocator.quarantine_count = 0
    allocator.invariant_counters = StagingInvariantCounters()
    allocator._quarantine_callback = None
    allocator.next_alloc_id = 0
    allocator.watermark_round = 0
    allocator.watermark_tail = 0
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
            kv_receiver=receiver,
            _staging_progress_lock=threading.Lock(),
            _staging_stall_failed=False,
            _staging_stall_since=None,
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
        self.assertFalse(handler.submit_last_scatter_async(19))

        self.assertTrue(lifecycle.terminal)
        handler.fail_staging_room.assert_called_once()
        handler._scatter_region.assert_not_called()

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
        self.assertTrue(handler.submit_last_scatter_async(4))
        handler.submit_chunk_scatter.assert_called_once_with(
            4, 0, 0, 2, is_last_chunk=True
        )

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

    def test_first_quarantine_arms_one_process_exit_timer(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.scheduler = SimpleNamespace(set_system_status=MagicMock())
        handler._quarantine_exit_lock = threading.Lock()
        handler._quarantine_exit_timer = None
        handler._fatal_shutdown = MagicMock()
        handler.staging_allocator = SimpleNamespace(quarantine_count=1)
        timers = []

        class FakeTimer:
            def __init__(self, delay, callback):
                self.delay = delay
                self.callback = callback
                self.daemon = False
                timers.append(self)

            def start(self):
                pass

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
            handler._on_first_quarantine(4, "test")
            handler._on_first_quarantine(5, "duplicate")

        self.assertEqual(len(timers), 1)
        self.assertEqual(timers[0].delay, 12.5)
        timers[0].callback()
        handler._fatal_shutdown.assert_called_once_with(
            -1, 1, reason="staging-quarantine"
        )

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
