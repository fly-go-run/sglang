import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVSender,
    _StagingSenderRoomState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlSenderFailureCleanup(unittest.TestCase):
    def _make_sender(self, room, expected_exc, room_state):
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._send_error = None
        staging_ctx = SimpleNamespace(
            prefetched_rooms={room, 8},
            prefetch_requested={(room, 0, "session-a"), (8, 0, "session-b")},
            send_ops={(room, 0, "session-a"): object()},
            send_ops_lock=threading.Lock(),
        )
        sender.kv_mgr = mgr = SimpleNamespace(
            enable_staging=True,
            _staging_ctx=staging_ctx,
            _sender_room_lock=threading.RLock(),
            _staging_sender_rooms={room: room_state},
            request_status={room: object()},
            req_to_decode_prefix_len={room: 3},
            transfer_infos={room: object()},
            exceptions={room: expected_exc},
            failure_records={room: "transfer failed"},
            failure_lock=threading.Lock(),
            _report_room_stopped=MagicMock(),
        )
        for name in (
            "_clear_staging_sender_room",
            "_clear_staging_send_ops",
            "_mark_staging_stop_pending",
            "_staging_stop_is_pending",
            "_pop_staging_stop_pending",
        ):
            setattr(mgr, name, getattr(NixlKVManager, name).__get__(mgr))
        return sender, mgr, staging_ctx

    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 7
        expected_exc = RuntimeError("transfer failed")
        sender, mgr, staging_ctx = self._make_sender(
            room, expected_exc, _StagingSenderRoomState()
        )

        with self.assertRaises(RuntimeError) as cm:
            sender.failure_exception()

        self.assertIs(cm.exception, expected_exc)
        self.assertTrue(sender._send_failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.exceptions)
        self.assertNotIn(room, sender.kv_mgr.failure_records)
        self.assertNotIn(room, sender.kv_mgr._staging_sender_rooms)
        self.assertNotIn(room, staging_ctx.prefetched_rooms)
        self.assertNotIn((room, 0, "session-a"), staging_ctx.prefetch_requested)
        self.assertIn(8, staging_ctx.prefetched_rooms)
        self.assertIn((8, 0, "session-b"), staging_ctx.prefetch_requested)
        self.assertFalse(staging_ctx.send_ops)
        # Nothing of the room was queued or in flight on this rank: the
        # cleanup itself tells decode this writer stopped.
        mgr._report_room_stopped.assert_called_once()
        self.assertFalse(mgr._staging_stop_is_pending(room))

    def test_failure_with_chunk_in_flight_defers_the_stop_report_to_the_worker(self):
        room = 9
        state = _StagingSenderRoomState()
        state.registered_chunks.add(0)  # queued/in flight, not completed
        sender, mgr, _ = self._make_sender(room, RuntimeError("x"), state)

        with self.assertRaises(RuntimeError):
            sender.failure_exception()

        mgr._report_room_stopped.assert_not_called()
        self.assertTrue(mgr._staging_stop_is_pending(room))
        self.assertNotIn(room, mgr.request_status)

    def test_success_clear_does_not_report(self):
        room = 11
        sender, mgr, _ = self._make_sender(room, None, None)
        mgr._staging_sender_rooms.clear()
        mgr.request_status[room] = KVPoll.Success
        sender.conclude_state = KVPoll.Success

        sender.clear()

        mgr._report_room_stopped.assert_not_called()
        self.assertFalse(mgr._staging_stop_is_pending(room))


if __name__ == "__main__":
    unittest.main()
