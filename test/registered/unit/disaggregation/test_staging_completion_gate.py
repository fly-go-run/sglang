"""GPU-free tests for the decode completion gate inside the NIXL receiver poll.

Transport done is not request done: the first-token metadata stamp must have
landed and, for hetero-TP rooms, the staged KV must be scattered. The gate
runs before the receiver caches Success, so its waiting timeout bounds the
whole path and no separate "demoted" clock is needed.
"""

import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.staging_buffer import StagingInvariantCounters
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    StagingRoomLifecycle,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.nixl.conn import NixlKVReceiver
from sglang.srt.disaggregation.utils import poll_and_all_reduce_with_staging
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 9


def _make_receiver(gate, snapshot=None, init_age_s=0.0, waiting_timeout=300):
    receiver = object.__new__(NixlKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        check_status=lambda room: KVPoll.WaitingForInput,
        update_transfer_status=MagicMock(),
        check_transfer_done=lambda room: True,
        addr_to_rooms_tracker=defaultdict(set),
        transfer_statuses={ROOM: object()},
        completion_gate=gate,
        completion_snapshot=snapshot,
        waiting_timeout=waiting_timeout,
        record_failure=MagicMock(),
        update_status=MagicMock(),
    )
    receiver.bootstrap_room = ROOM
    receiver.bootstrap_addr = "10.0.0.1:8868"
    receiver.conclude_state = None
    receiver.started_transfer = True
    receiver.init_time = time.time() - init_age_s
    receiver.abort_notified = True
    receiver.bootstrap_infos = None
    receiver._completion_gate_blocked = False
    return receiver


def _make_staging_handler(scatter_done):
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
        require_staging=True,
        chunk_staging_infos=[(-1, -1, 0, -1, 0)],
        conclude_state=None,
        poll=lambda: KVPoll.Success,
    )
    decode_req = SimpleNamespace(
        req=SimpleNamespace(
            bootstrap_room=ROOM, rid="rid-9", bootstrap_host="10.0.0.1"
        ),
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
    )
    lifecycle = StagingRoomLifecycle(
        chunk_states={0: "SCATTER_DONE" if scatter_done else "WRITABLE"},
        chunk_geometry={0: (0, 4)},
    )
    if not scatter_done:
        receiver.chunk_staging_infos = [(11, 256, 0, 512, 4)]
    handler._room_to_decode_req = {ROOM: decode_req}
    handler._room_to_receiver = {ROOM: receiver}
    handler._room_lifecycles = {ROOM: lifecycle}
    handler._free_and_send_watermark = MagicMock()
    return handler, decode_req, receiver


def _make_queue(handler, decode_req, metadata_stamp):
    queue = object.__new__(DecodeTransferQueue)
    queue.queue = [decode_req]
    queue.staging_handler = handler
    queue.metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.tensor([[metadata_stamp]], dtype=torch.int64)
    )
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_transfer_backend="nixl")
    )
    return queue


class TestReceiverCompletionGate(CustomTestCase):
    def test_gate_blocks_success_without_caching(self):
        gate = MagicMock(return_value=False)
        receiver = _make_receiver(gate)
        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertIsNone(receiver.conclude_state)
        self.assertIn(ROOM, receiver.kv_mgr.transfer_statuses)
        self.assertTrue(receiver._completion_gate_blocked)
        gate.assert_called_once_with(ROOM)

        gate.return_value = True
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertEqual(receiver.conclude_state, KVPoll.Success)
        self.assertNotIn(ROOM, receiver.kv_mgr.transfer_statuses)
        self.assertFalse(receiver._completion_gate_blocked)

    def test_waiting_timeout_still_bounds_a_gated_room(self):
        snapshot = MagicMock(return_value="snap")
        receiver = _make_receiver(
            MagicMock(return_value=False), snapshot=snapshot, waiting_timeout=1
        )
        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        receiver.init_time = time.time() - 5
        with patch("sglang.srt.disaggregation.nixl.conn.logger") as log:
            self.assertEqual(receiver.poll(), KVPoll.Failed)
        receiver.kv_mgr.update_status.assert_called_once_with(ROOM, KVPoll.Failed)
        snapshot.assert_called_once_with(ROOM)
        self.assertTrue(
            any(
                "STAGING_COMPLETION_TIMEOUT" in str(c) for c in log.error.call_args_list
            )
        )

    def test_no_gate_keeps_legacy_behaviour(self):
        receiver = _make_receiver(None)
        self.assertEqual(receiver.poll(), KVPoll.Success)
        self.assertNotIn(ROOM, receiver.kv_mgr.transfer_statuses)


class TestTransferQueueCompletionGate(CustomTestCase):
    def test_metadata_stamp_missing_blocks(self):
        handler, decode_req, _ = _make_staging_handler(scatter_done=True)
        queue = _make_queue(handler, decode_req, metadata_stamp=0)
        self.assertFalse(queue._completion_gate(ROOM))
        self.assertIn("metadata_room=0", queue._completion_snapshot(ROOM))

    def test_staging_not_scattered_blocks_and_records_success(self):
        handler, decode_req, _ = _make_staging_handler(scatter_done=False)
        queue = _make_queue(handler, decode_req, metadata_stamp=ROOM)
        self.assertFalse(queue._completion_gate(ROOM))
        # Transport done is the authoritative all-ranks Success.
        self.assertTrue(decode_req._staging_all_success)
        self.assertFalse(decode_req._staging_scatter_done)

    def test_releases_once_scatter_done_and_metadata_landed(self):
        handler, decode_req, _ = _make_staging_handler(scatter_done=True)
        queue = _make_queue(handler, decode_req, metadata_stamp=ROOM)
        self.assertTrue(queue._completion_gate(ROOM))
        self.assertTrue(decode_req._staging_all_success)
        self.assertTrue(decode_req._staging_scatter_done)

    def test_fake_transfer_skips_metadata_stamp(self):
        handler, decode_req, receiver = _make_staging_handler(scatter_done=True)
        decode_req.req.bootstrap_host = "2.2.2.2"
        receiver.require_staging = False
        queue = _make_queue(handler, decode_req, metadata_stamp=0)
        self.assertTrue(queue._completion_gate(ROOM))

    def test_unknown_room_does_not_block(self):
        handler, decode_req, _ = _make_staging_handler(scatter_done=True)
        queue = _make_queue(handler, decode_req, metadata_stamp=0)
        self.assertTrue(queue._completion_gate(ROOM + 1))


class TestStagingPollWithGate(CustomTestCase):
    def test_gated_backend_skips_poll_time_demotion(self):
        handler, decode_req, receiver = _make_staging_handler(scatter_done=False)
        handler.kv_manager.completion_gate = MagicMock()
        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"):
            polls = poll_and_all_reduce_with_staging([decode_req], handler, None)
        # The receiver already applied the gate; utils must not second-guess it.
        self.assertEqual(polls, [int(KVPoll.Success)])

    def test_ungated_backend_still_demotes(self):
        handler, decode_req, receiver = _make_staging_handler(scatter_done=False)
        handler.kv_manager.completion_gate = None
        with patch("sglang.srt.disaggregation.utils.dist.all_reduce"):
            polls = poll_and_all_reduce_with_staging([decode_req], handler, None)
        self.assertEqual(polls, [int(KVPoll.Transferring)])
        self.assertTrue(decode_req._staging_all_success)


if __name__ == "__main__":
    import unittest

    unittest.main()
