import ctypes
import threading
from collections import defaultdict

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.shm_pinned.conn import (
    ShmPinnedKVManager,
    ShmPinnedKVSender,
    TransferRequest,
)
from sglang.srt.disaggregation.shm_pinned.transfer_engine import (
    ShmPinnedTransferEngine,
)
from sglang.srt.disaggregation.shm_pinned.utils import (
    ShmPinnedInfo,
    SlotMeta,
    SlotState,
    calculate_slot_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_bare_manager() -> ShmPinnedKVManager:
    """Build a manager with only the fields these tests exercise.

    Bypasses __init__ so we don't pull in bootstrap / engine setup.
    """
    mgr = object.__new__(ShmPinnedKVManager)
    mgr.pending_requests = {}
    mgr.room_sessions = defaultdict(set)
    mgr.latest_session_for_room = {}
    mgr.pending_lock = threading.Lock()
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.session_engines = {}
    return mgr


def _make_bare_sender(mgr: ShmPinnedKVManager, room: int) -> ShmPinnedKVSender:
    sender = object.__new__(ShmPinnedKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = room
    sender.session_id = None
    sender.session_engine = None
    sender.dst_kv_indices = None
    return sender


def test_slot_meta_round_trip():
    meta = SlotMeta(
        state=SlotState.READY,
        room=17,
        index_start=3,
        index_len=7,
        is_last=1,
        valid_bytes=192,
        seqno=9,
        owner_pid=1234,
    )

    assert SlotMeta.unpack(meta.pack()) == meta


def test_shm_pinned_info_round_trip():
    info = ShmPinnedInfo(
        data_shm_name="/pytest_data",
        meta_shm_name="/pytest_meta",
        sem_free_name="/pytest_free",
        sem_ready_name="/pytest_ready",
        sem_slot_name="/pytest_slot",
        slot_count=4,
        slot_bytes=256,
        session_id="session-1",
        kv_item_lens=[64, 64, 64, 64],
        decode_host="127.0.0.1",
        decode_port=9001,
    )

    assert ShmPinnedInfo.from_dict(info.to_dict()) == info


def test_calculate_slot_bytes():
    assert calculate_slot_bytes(4, [8, 12], extra_slot_bytes=16) == 96

    with pytest.raises(ValueError, match="kv_item_lens"):
        calculate_slot_bytes(1, [])


def test_transfer_engine_cpu_smoke(monkeypatch):
    pytest.importorskip("posix_ipc")

    def cpu_memcpy(self, dst_ptr: int, src_ptr: int, num_bytes: int) -> None:
        ctypes.memmove(dst_ptr, src_ptr, num_bytes)

    monkeypatch.setattr(ShmPinnedTransferEngine, "cuda_memcpy", cpu_memcpy)

    kv_item_lens = [4, 8]
    num_pages = 2
    aux_bytes = b"AUX123"
    kv_total_bytes = num_pages * sum(kv_item_lens)

    src0 = ctypes.create_string_buffer(b"ABCDWXYZ", 8)
    src1 = ctypes.create_string_buffer(b"abcdefgh12345678", 16)
    aux_src = ctypes.create_string_buffer(aux_bytes, len(aux_bytes))

    dst0 = ctypes.create_string_buffer(8)
    dst1 = ctypes.create_string_buffer(16)
    aux_dst = ctypes.create_string_buffer(len(aux_bytes))

    decode_engine = ShmPinnedTransferEngine(
        session_id="pytest-smoke",
        gpu_id=0,
        slot_count=2,
        chunk_pages=num_pages,
        kv_item_lens=kv_item_lens,
        extra_slot_bytes=len(aux_bytes),
        create=True,
    )
    prefill_engine = ShmPinnedTransferEngine(
        session_id="pytest-smoke",
        gpu_id=0,
        create=False,
    )

    try:
        prefill_engine.open_from_info(decode_engine.export_info())
        decode_engine._data_ptr = ctypes.addressof(
            ctypes.c_ubyte.from_buffer(decode_engine.data_mmap)
        )
        prefill_engine._data_ptr = ctypes.addressof(
            ctypes.c_ubyte.from_buffer(prefill_engine.data_mmap)
        )

        slot_idx = prefill_engine.wait_free(timeout=1.0)
        slot_ptr = prefill_engine.get_slot_data_ptr(slot_idx)

        prefill_engine.cuda_memcpy(slot_ptr, ctypes.addressof(src0), len(src0.raw))
        prefill_engine.cuda_memcpy(
            slot_ptr + len(src0.raw),
            ctypes.addressof(src1),
            len(src1.raw),
        )
        ctypes.memmove(
            slot_ptr + kv_total_bytes,
            ctypes.addressof(aux_src),
            len(aux_bytes),
        )

        prefill_engine.write_meta(
            slot_idx=slot_idx,
            room=17,
            index_start=0,
            index_len=num_pages,
            is_last=True,
            valid_bytes=kv_total_bytes + len(aux_bytes),
            seqno=1,
        )
        prefill_engine.post_ready(slot_idx)

        ready_idx = decode_engine.wait_ready(timeout=1.0)
        meta = decode_engine.read_meta(ready_idx)
        assert meta.room == 17
        assert meta.index_len == num_pages
        assert meta.is_last == 1

        slot_ptr = decode_engine.get_slot_data_ptr(ready_idx)
        decode_engine.cuda_memcpy(
            ctypes.addressof(dst0),
            slot_ptr,
            len(dst0.raw),
        )
        decode_engine.cuda_memcpy(
            ctypes.addressof(dst1),
            slot_ptr + len(dst0.raw),
            len(dst1.raw),
        )
        ctypes.memmove(
            ctypes.addressof(aux_dst),
            slot_ptr + kv_total_bytes,
            len(aux_bytes),
        )
        decode_engine.post_free(ready_idx)

        assert dst0.raw == src0.raw
        assert dst1.raw == src1.raw
        assert aux_dst.raw == aux_bytes
    finally:
        prefill_engine.close()
        decode_engine.close()


def test_sender_attach_binds_latest_session_and_resets_room_state():
    """A new session on a reused room must override a prior terminal state
    and the sender fallback attach must resolve to the latest session."""
    mgr = _make_bare_manager()
    room = 17

    old_engine = object()
    old_req = TransferRequest(
        room=room,
        session_id="old",
        dst_kv_indices=np.array([1, 2], dtype=np.int32),
        engine=old_engine,
        total_pages=2,
    )
    mgr._add_pending_request(old_req)
    # Simulate the old session terminating with Failed before it is cleaned up.
    mgr.record_failure(room, "old failure")
    mgr.update_status(room, KVPoll.Failed)
    assert mgr.get_status(room) == KVPoll.Failed
    assert mgr.get_failure_message(room) == "old failure"

    new_engine = object()
    new_req = TransferRequest(
        room=room,
        session_id="new",
        dst_kv_indices=np.array([3, 4], dtype=np.int32),
        engine=new_engine,
        total_pages=2,
    )
    mgr._add_pending_request(new_req)

    # Stale entry must have been GC'd when the new session took the room.
    assert mgr.get_pending_request("old", room) is None
    assert mgr.get_latest_pending_request_for_room(room) is new_req
    # Room state reset so the new session starts fresh.
    assert mgr.request_status[room] == KVPoll.Bootstrapping
    assert mgr.get_failure_message(room) is None

    # Sender without a bound session_id must resolve to the latest session.
    sender = _make_bare_sender(mgr, room)
    assert sender._ensure_session_ready() is True
    assert sender.session_id == "new"
    assert sender.session_engine is new_engine
    assert np.array_equal(sender.dst_kv_indices, new_req.dst_kv_indices)


def test_set_room_state_if_latest_blocks_stale_writes():
    """ABORT/FAIL/Success from a stale session must not overwrite the
    latest session's room status."""
    mgr = _make_bare_manager()
    room = 23

    old_req = TransferRequest(
        room=room,
        session_id="old",
        dst_kv_indices=np.array([1], dtype=np.int32),
        engine=object(),
        total_pages=1,
    )
    new_req = TransferRequest(
        room=room,
        session_id="new",
        dst_kv_indices=np.array([2], dtype=np.int32),
        engine=object(),
        total_pages=1,
    )
    mgr._add_pending_request(old_req)
    mgr._add_pending_request(new_req)
    # "new" owns the room now; move it to WaitingForInput.
    assert (
        mgr.set_room_state_if_latest(room, "new", KVPoll.WaitingForInput) is True
    )
    assert mgr.get_status(room) == KVPoll.WaitingForInput

    # Stale Failed from "old" must be ignored.
    assert mgr.set_room_state_if_latest(room, "old", KVPoll.Failed, "stale") is False
    assert mgr.get_status(room) == KVPoll.WaitingForInput
    assert mgr.get_failure_message(room) is None

    # Stale Success from "old" must also be ignored (otherwise max() would
    # let Success leak through and poison the latest session's poll view).
    assert mgr.set_room_state_if_latest(room, "old", KVPoll.Success) is False
    assert mgr.get_status(room) == KVPoll.WaitingForInput

    # Latest session is still allowed to advance.
    assert mgr.set_room_state_if_latest(room, "new", KVPoll.Success) is True
    assert mgr.get_status(room) == KVPoll.Success


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
