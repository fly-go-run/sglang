"""GPU-free tests for the decode-side demoted-Success timeout.

A room whose NIXL transfer concluded Success can be held in Transferring by
the staging ``is_done`` gate or the metadata gate. Neither gate has a timeout
and the receiver stops running its waiting timeout once Success is cached, so
without this clock the request parks in the transfer queue forever.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.staging_buffer import StagingInvariantCounters
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    StagingRoomLifecycle,
)
from sglang.srt.disaggregation.utils import poll_and_all_reduce_with_staging
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 9
MONOTONIC = "sglang.srt.disaggregation.common.staging_handler.time.monotonic"


def _make_handler(chunk_infos=None):
    handler = object.__new__(DecodeStagingHandler)
    handler.decode_tp = 1
    handler.tp_rank = 0
    handler.kv_manager = MagicMock()
    handler.kv_manager.transfer_statuses = {
        ROOM: SimpleNamespace(
            received_aux=True,
            expected_kvs_per_pp={0: 1},
            received_kvs_per_pp={0: {0}},
        )
    }
    handler.invariant_counters = StagingInvariantCounters()
    handler.staging_allocator = SimpleNamespace(
        _scatter_stream=object(), allocations={}
    )
    receiver = SimpleNamespace(
        require_staging=True,
        chunk_staging_infos=(
            [(-1, -1, 0, -1, 0)] if chunk_infos is None else list(chunk_infos)
        ),
        conclude_state=KVPoll.Success,
        poll=lambda: KVPoll.Success,
    )
    decode_req = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=ROOM, rid="rid-9"),
        kv_receiver=receiver,
        metadata_buffer_index=0,
        _staging_progress_lock=threading.Lock(),
        _staging_stall_failed=False,
        _staging_stall_since=None,
        _staging_data_started=True,
        _staging_scatter_done=False,
        _staging_all_success=False,
        _staging_success_ts=0.0,
        _chunk_events=[],
        _staging_demoted_since=None,
    )
    handler._room_to_decode_req = {ROOM: decode_req}
    handler._room_to_receiver = {ROOM: receiver}
    handler._room_lifecycles = {
        ROOM: StagingRoomLifecycle(
            chunk_states={0: "SCATTER_DONE"}, chunk_geometry={0: (0, 4)}
        )
    }
    return handler, decode_req, receiver


class TestStagingDemotedTimeout(CustomTestCase):
    def test_not_demoted_resets_clock(self):
        handler, decode_req, _ = _make_handler()
        decode_req._staging_demoted_since = 12.0
        self.assertFalse(handler.check_demoted_timeout(decode_req, False, False))
        self.assertIsNone(decode_req._staging_demoted_since)
        handler.kv_manager.update_status.assert_not_called()

    def test_demoted_without_allocations_fails_after_timeout(self):
        handler, decode_req, receiver = _make_handler()
        with patch(MONOTONIC, return_value=100.0):
            self.assertFalse(handler.check_demoted_timeout(decode_req, True, False))
        self.assertEqual(decode_req._staging_demoted_since, 100.0)
        with patch(MONOTONIC, return_value=130.0):
            self.assertFalse(handler.check_demoted_timeout(decode_req, True, False))
        self.assertEqual(receiver.conclude_state, KVPoll.Success)
        handler.kv_manager.update_status.assert_not_called()

        with patch(MONOTONIC, return_value=161.0):
            self.assertTrue(
                handler.check_demoted_timeout(decode_req, True, False, None)
            )
        handler.kv_manager.update_status.assert_called_once_with(ROOM, KVPoll.Failed)
        room, reason = handler.kv_manager.record_failure.call_args.args
        self.assertEqual(room, ROOM)
        self.assertIn("STAGING_DEMOTED_TIMEOUT", reason)
        self.assertIn("staging_gate=True", reason)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertIsNone(decode_req._staging_demoted_since)
        # Bounding a demoted room must never touch the quarantine path.
        self.assertEqual(handler.staging_allocator.allocations, {})

    def test_metadata_gate_alone_is_bounded(self):
        handler, decode_req, receiver = _make_handler()
        receiver.require_staging = False
        with patch(MONOTONIC, return_value=0.0):
            self.assertFalse(handler.check_demoted_timeout(decode_req, False, True, 0))
        with patch(MONOTONIC, return_value=61.0):
            self.assertTrue(handler.check_demoted_timeout(decode_req, False, True, 0))
        _, reason = handler.kv_manager.record_failure.call_args.args
        self.assertIn("metadata_gate=True", reason)

    def test_room_holding_allocations_is_left_to_stall_watchdog(self):
        handler, decode_req, _ = _make_handler(chunk_infos=[(3, 0, 0, 4, 4)])
        with patch(MONOTONIC, return_value=0.0):
            self.assertFalse(handler.check_demoted_timeout(decode_req, True, False))
        self.assertIsNone(decode_req._staging_demoted_since)
        with patch(MONOTONIC, return_value=1000.0):
            self.assertFalse(handler.check_demoted_timeout(decode_req, True, False))
        handler.kv_manager.update_status.assert_not_called()

    @staticmethod
    def _metadata_gate(stamp):
        """Metadata gate inputs: bootstrap_room stamp 0 keeps the room demoted."""
        buffers = SimpleNamespace(
            bootstrap_room=torch.tensor([[stamp]], dtype=torch.int64)
        )
        server_args = SimpleNamespace(disaggregation_transfer_backend="nixl")
        return buffers, server_args

    def test_poll_path_reports_failed_after_timeout(self):
        """The metadata gate has no clock of its own; the demoted timer bounds it."""
        handler, decode_req, receiver = _make_handler()
        decode_req.req.bootstrap_host = "10.0.0.1"
        buffers, server_args = self._metadata_gate(0)
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce"
        ) as all_reduce, patch(MONOTONIC, return_value=0.0):
            polls = poll_and_all_reduce_with_staging(
                [decode_req], handler, None, buffers, server_args
            )
        self.assertEqual(polls, [int(KVPoll.Transferring)])
        # Staging itself completed from the raw Success observation.
        self.assertTrue(decode_req._staging_all_success)
        self.assertTrue(handler.is_done(decode_req))
        self.assertEqual(decode_req._staging_demoted_since, 0.0)
        all_reduce.assert_called_once()

        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"), patch(
            MONOTONIC, return_value=61.0
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req], handler, None, buffers, server_args
            )
        self.assertEqual(polls, [int(KVPoll.Failed)])
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        handler.kv_manager.update_status.assert_called_once_with(ROOM, KVPoll.Failed)
        _, reason = handler.kv_manager.record_failure.call_args.args
        self.assertIn("metadata_gate=True", reason)

    def test_poll_path_clears_clock_once_gate_releases(self):
        handler, decode_req, _ = _make_handler()
        decode_req.req.bootstrap_host = "10.0.0.1"
        buffers, server_args = self._metadata_gate(0)
        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"), patch(
            MONOTONIC, return_value=0.0
        ):
            poll_and_all_reduce_with_staging(
                [decode_req], handler, None, buffers, server_args
            )
        self.assertEqual(decode_req._staging_demoted_since, 0.0)
        # The metadata stamp lands: the gate releases and the clock is cleared.
        buffers.bootstrap_room[0, 0] = ROOM
        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"), patch(
            MONOTONIC, return_value=61.0
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req], handler, None, buffers, server_args
            )
        self.assertEqual(polls, [int(KVPoll.Success)])
        self.assertIsNone(decode_req._staging_demoted_since)
        handler.kv_manager.update_status.assert_not_called()

    def test_poll_path_records_success_when_notification_trigger_missed(self):
        """Raw Success with all_success never recorded (non-last chunk landed last)."""
        handler, decode_req, receiver = _make_handler()
        # Restore the real handler methods the fixture leaves unset.
        handler._free_and_send_watermark = MagicMock()
        self.assertFalse(decode_req._staging_all_success)
        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"), patch(
            MONOTONIC, return_value=0.0
        ):
            polls = poll_and_all_reduce_with_staging([decode_req], handler, None)
        self.assertTrue(decode_req._staging_all_success)
        self.assertTrue(decode_req._staging_scatter_done)
        self.assertEqual(polls, [int(KVPoll.Success)])
        self.assertIsNone(decode_req._staging_demoted_since)


if __name__ == "__main__":
    import unittest

    unittest.main()
