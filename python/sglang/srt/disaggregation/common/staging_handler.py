"""
Staging handler for heterogeneous TP KV cache transfer.

Isolates staging scatter lifecycle from decode.py and conn.py.
Generic (backend-agnostic) code is at the top; mooncake-specific
protocol code is at the bottom.
"""

from __future__ import annotations

import dataclasses
from collections import deque
import logging
import os
import random
import signal
import struct
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Avoid a tight dequeue/requeue loop while the prefill waits for the decode
# ring watermark to advance.
STAGING_WATERMARK_WAIT_S = 0.001

# Decode must never synchronously send control-plane messages from the scheduler
# thread. A dead or backpressured prefill peer can otherwise freeze decode and
# every connected prefill group. The sender below coalesces each subscriber to
# its newest watermark and retries outside the scheduler thread.

# The staging ring is shared by every prefill session: one room that holds
# allocations without making scatter progress pins the global watermark and
# starves all sessions (observed 2026-08-20: remote_wm stuck for 400+s, both
# prefills frozen, decode then killed by its watchdog). Fail such a room after
# this many seconds so the existing Failed -> unregister -> release_room path
# returns its allocations, well before any pod-level watchdog acts. The clock
# only runs while the room actually holds allocations, so requests that are
# merely queued behind a busy prefill are not affected.
STAGING_ROOM_STALL_TIMEOUT_S = float(
    os.environ.get("SGLANG_DISAGG_STAGING_STALL_TIMEOUT_S", "60")
)
STAGING_SCATTER_DRAIN_TIMEOUT_S = float(
    os.environ.get("SGLANG_DISAGG_STAGING_SCATTER_DRAIN_TIMEOUT_S", "5")
)

# Quarantine isolates ring memory whose writer cannot be proven stopped; that
# alone is safe, so a quarantine caused by a stall, a failed transfer or a
# room released with live allocations only costs capacity. Process-level
# fail-stop (NotReady + exit, Kubernetes rebuilds clean memory) is reserved
# for two cases: an invariant violation that says memory may already be
# corrupted, or quarantined capacity exceeding this fraction of the ring.
STAGING_QUARANTINE_EXIT_FRACTION = float(
    os.environ.get("SGLANG_DISAGG_STAGING_QUARANTINE_EXIT_FRACTION", "0.10")
)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest


# ======================================================================
# Generic staging state and handler (backend-agnostic)
# ======================================================================


@dataclasses.dataclass
class DecodeStagingContext:
    """Staging-specific context for decode mode."""

    allocator: object = None
    room_bootstrap: dict = dataclasses.field(default_factory=dict)
    room_receivers: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PrefillStagingContext:
    """Staging-specific context for prefill mode."""

    buffers: list = dataclasses.field(default_factory=list)
    remote_watermarks: dict = dataclasses.field(default_factory=dict)
    watermark_cv: threading.Condition = dataclasses.field(
        default_factory=threading.Condition
    )
    # (room, chunk_idx, session_id) keys for chunks already requested.
    prefetch_requested: set = dataclasses.field(default_factory=set)
    # Rooms that have already had their full prefetch fan-out triggered. Used
    # to short-circuit per-room prefetch entry on every chunk after the first.
    prefetched_rooms: set = dataclasses.field(default_factory=set)
    prefetch_sockets: dict = dataclasses.field(default_factory=dict)
    # Diagnostic timestamps for a chunk waiting on the remote ring watermark.
    watermark_wait_started: dict = dataclasses.field(default_factory=dict)
    watermark_wait_last_log: dict = dataclasses.field(default_factory=dict)
    # Last remote watermark observed per waiting chunk: while the watermark
    # keeps advancing the wait is backlog, not a stall, and the stall clock
    # restarts from the latest advance.
    watermark_wait_last_wm: dict = dataclasses.field(default_factory=dict)
    # Persistent per-(room, chunk, destination) send state. Entries survive
    # defer/requeue and are removed only when the room is cleared.
    send_ops: dict = dataclasses.field(default_factory=dict)
    send_ops_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    invariant_counters: object = None


@dataclasses.dataclass
class StagingRoomLifecycle:
    """Decode-side terminal fence and persistent per-chunk tombstones."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    terminal: bool = False
    quarantined: bool = False
    chunk_states: dict = dataclasses.field(default_factory=dict)
    chunk_geometry: dict = dataclasses.field(default_factory=dict)
    seen_writer_slots: dict = dataclasses.field(default_factory=dict)
    writer_intervals: dict = dataclasses.field(default_factory=dict)


class DecodeStagingHandler:
    """Decode-side staging scatter lifecycle manager.

    Scatter submission can be called from the decode_thread (background) as
    soon as all writers/ranks have arrived, while event checking and freeing
    always run on the scheduler main thread.
    """

    def __init__(
        self,
        kv_manager,
        staging_allocator,
        kv_buffer_info: dict,
        decode_tp: int,
        total_kv_heads: int,
        tp_rank: int,
        scheduler,
        fatal_shutdown=None,
    ):
        self.kv_manager = kv_manager
        self.staging_allocator = staging_allocator
        self.kv_buffer_info = kv_buffer_info
        self.decode_tp = decode_tp
        self.total_kv_heads = total_kv_heads
        self.tp_rank = tp_rank
        self.scheduler = scheduler
        self._fatal_shutdown = fatal_shutdown
        self._room_to_decode_req: dict = {}
        # Failure paths clear decode_req.kv_receiver before unregistering the
        # room, so retain the receiver needed to release staging allocations.
        self._room_to_receiver: dict = {}
        self._room_lifecycles: dict[int, StagingRoomLifecycle] = {}
        self.invariant_counters = staging_allocator.invariant_counters
        self._quarantine_exit_lock = threading.Lock()
        self._quarantine_exit_timer = None
        self.staging_allocator.set_quarantine_callback(self._on_quarantine)
        self.staging_allocator._staging_handler = self
        # STAGING_REQs that found no free extent wait here in FIFO order and
        # are granted (and answered with STAGING_RSP) as releases free space.
        # This is the staging layer's admission control; no watermark exists.
        self._pending_allocs: deque = deque()
        self._pending_lock = threading.Lock()

    # Watermark subscriptions no longer exist: a granted extent is exclusive
    # until released, so prefill never waits on decode. Kept as no-ops for the
    # shared call sites in the NIXL/Mooncake managers.
    def register_wm_subscriber(self, receiver, session_id: str) -> None:
        return None

    def snapshot_wm_subscribers(self, bootstrap_info_groups) -> list:
        return []

    def unregister_wm_subscribers(self, subscriber_tokens, reason: str) -> None:
        return None

    # ------------------------------------------------------------------
    # Pending allocations: FIFO admission for STAGING_REQs that found no extent
    # ------------------------------------------------------------------

    def enqueue_pending_alloc(
        self, room: int, chunk_idx: int, required: int, session_id: str, pages: int
    ) -> None:
        with self._pending_lock:
            for item in self._pending_allocs:
                if item[0] == room and item[1] == chunk_idx:
                    return
            self._pending_allocs.append((room, chunk_idx, required, session_id, pages))
            logger.warning(
                "[STAGING_ALLOC_PENDING] room=%s chunk=%s bytes=%s queued=%s "
                "free=%s largest=%s",
                room,
                chunk_idx,
                required,
                len(self._pending_allocs),
                self.staging_allocator.free_bytes(),
                self.staging_allocator.largest_extent(),
            )

    def drop_pending_allocs(self, room: int) -> None:
        with self._pending_lock:
            kept = [item for item in self._pending_allocs if item[0] != room]
            if len(kept) != len(self._pending_allocs):
                self._pending_allocs = deque(kept)

    def pending_alloc_count(self) -> int:
        with self._pending_lock:
            return len(self._pending_allocs)

    def _service_pending_allocs(self) -> None:
        """Grant queued STAGING_REQs in FIFO order while extents fit.

        Stops at the first request that still does not fit so a large chunk is
        never starved by smaller ones queued behind it.
        """
        # Handlers built without __init__ (unit fixtures) have no queue.
        if not getattr(self, "_pending_allocs", None):
            return
        while True:
            with self._pending_lock:
                if not self._pending_allocs:
                    return
                room, chunk_idx, required, session_id, pages = self._pending_allocs[0]
                lifecycle = self._room_lifecycles.get(room)
                receiver = self._room_to_receiver.get(room)
                if lifecycle is None or receiver is None or lifecycle.terminal:
                    self._pending_allocs.popleft()
                    continue
                attainable = self.staging_allocator.max_attainable_extent()
                if required > attainable:
                    # A quarantine after this request was queued shrank the
                    # widest attainable extent below it: it can never be
                    # granted and must not block the queue head.
                    self._pending_allocs.popleft()
                    unattainable = (room, chunk_idx, required, attainable)
                    result = None
                else:
                    unattainable = None
                    result = self.staging_allocator.assign(required)
                    if result is None:
                        return
                    self._pending_allocs.popleft()
            if unattainable is not None:
                room, chunk_idx, required, attainable = unattainable
                self.fail_staging_room(
                    room,
                    f"[STAGING_ALLOC_UNATTAINABLE] room={room} chunk={chunk_idx} "
                    f"needs {required} bytes but the largest attainable extent "
                    f"is {attainable} bytes after quarantine",
                )
                continue
            alloc_id, offset, _ = result
            granted = False
            with lifecycle.lock:
                infos = getattr(receiver, "chunk_staging_infos", [])
                state = lifecycle.chunk_states.get(chunk_idx)
                if not lifecycle.terminal and state == "PENDING_ALLOC":
                    while len(infos) <= chunk_idx:
                        infos.append((-1, -1, 0, -1, 0))
                    infos[chunk_idx] = (alloc_id, offset, 0, offset + required, pages)
                    lifecycle.chunk_states[chunk_idx] = "WRITABLE"
                    granted = True
            if not granted:
                self.staging_allocator.free(alloc_id)
                continue
            send_staging_rsp(
                receiver,
                self.kv_manager._staging_ctx.room_bootstrap,
                room,
                chunk_idx,
                offset,
                0,
                offset + required,
                session_id,
            )

    def num_writers_for(self, decode_req) -> int:
        """Compute num_writers for a specific request based on its prefill TP."""
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        if prefill_tp > self.decode_tp:
            return prefill_tp // max(1, self.decode_tp)
        return 1

    @staticmethod
    def _chunk_coverage_complete_locked(
        lifecycle: StagingRoomLifecycle,
        chunk_idx: int,
        num_writers: int,
    ) -> bool:
        expected_start, expected_pages = lifecycle.chunk_geometry[chunk_idx]
        expected_end = expected_start + expected_pages
        writer_intervals = lifecycle.writer_intervals.get(chunk_idx, {})
        if len(writer_intervals) != num_writers:
            return False
        for intervals in writer_intervals.values():
            cursor = expected_start
            for start, end in sorted(intervals):
                if start != cursor:
                    return False
                cursor = end
            if cursor != expected_end:
                return False
        return True

    def get_chunk_geometry(self, room: int, chunk_idx: int):
        """Return immutable expected geometry under the lifecycle lock."""
        lifecycle = self._room_lifecycles.get(room)
        if lifecycle is None:
            return None
        with lifecycle.lock:
            return lifecycle.chunk_geometry.get(chunk_idx)

    @classmethod
    def create(cls, kv_manager, scheduler, tp_rank: int) -> DecodeStagingHandler:
        """Factory: create handler. Raises if staging infra is missing."""
        staging_allocator = kv_manager._staging_ctx.allocator
        if staging_allocator is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager._staging_ctx.allocator is None. "
                "Check that the transfer backend correctly initializes the "
                "staging allocator."
            )
        kv_buffer_info = kv_manager.kv_buffer_tensors
        if kv_buffer_info is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager.kv_buffer_tensors is None. "
                "Check that set_kv_buffer_tensors() was called during kv_manager init."
            )
        decode_tp = kv_manager.attn_tp_size

        from sglang.srt.disaggregation.common.staging_buffer import (
            resolve_total_kv_heads,
        )

        total_kv_heads = resolve_total_kv_heads(kv_manager.kv_args, decode_tp)
        return cls(
            kv_manager=kv_manager,
            staging_allocator=staging_allocator,
            kv_buffer_info=kv_buffer_info,
            decode_tp=decode_tp,
            total_kv_heads=total_kv_heads,
            tp_rank=tp_rank,
            scheduler=scheduler,
            fatal_shutdown=getattr(
                kv_manager, "_request_fatal_transfer_shutdown", None
            ),
        )

    # ------------------------------------------------------------------
    # Registration: called from main thread (DecodeTransferQueue)
    # ------------------------------------------------------------------

    def _set_dynamo_notready_best_effort(self) -> None:
        """Use an integration-provided Dynamo status hook when one exists."""
        for name in ("set_dynamo_system_status", "set_system_status"):
            setter = getattr(self.scheduler, name, None)
            if not callable(setter):
                continue
            try:
                setter("notready")
                logger.error(
                    "[STAGING_QUARANTINE] Dynamo system status set to notready"
                )
                return
            except Exception:
                logger.exception(
                    "[STAGING_QUARANTINE] failed to set Dynamo system status "
                    "notready via %s",
                    name,
                )
        logger.warning(
            "[STAGING_QUARANTINE] no Dynamo system-status hook is available; "
            "continuing with process-level exit"
        )

    @staticmethod
    def _quarantine_is_invariant_violation(reason: str) -> bool:
        """Reasons tagged [STAGING_*] come from geometry/writer/lifecycle
        fences: memory may already hold foreign data. Plain reasons
        (staging-stall, decode-transfer-failed, room-release) are wedges or
        failures where isolating the allocation is sufficient."""
        return reason.startswith("[STAGING_")

    def _on_quarantine(self, alloc_id: int, reason: str) -> None:
        """Isolate by default; escalate to process fail-stop only when needed."""
        allocator = self.staging_allocator
        if self._quarantine_is_invariant_violation(reason):
            self._arm_process_exit(alloc_id, reason)
            return
        total = max(1, getattr(allocator, "total_size", 1))
        quarantined = allocator.quarantined_bytes()
        fraction = quarantined / total
        if fraction > STAGING_QUARANTINE_EXIT_FRACTION:
            self._arm_process_exit(
                alloc_id,
                f"staging-quarantine-capacity fraction={fraction:.3f} "
                f"quarantined_bytes={quarantined} total={total} last_reason={reason}",
            )
            return
        logger.error(
            "[STAGING_QUARANTINE_ISOLATED] alloc_id=%s reason=%s "
            "quarantine_count=%s quarantined_bytes=%s total=%s fraction=%.3f "
            "(below exit fraction %.3f; serving continues)",
            alloc_id,
            reason,
            allocator.quarantine_count,
            quarantined,
            total,
            fraction,
            STAGING_QUARANTINE_EXIT_FRACTION,
        )

    def _arm_process_exit(self, alloc_id: int, reason: str) -> None:
        """Mark notready and arm exactly one process-level fail-stop timer."""
        with self._quarantine_exit_lock:
            if self._quarantine_exit_timer is not None:
                return
            self._set_dynamo_notready_best_effort()
            delay_s = random.uniform(0.0, 60.0)

            def request_exit():
                logger.critical(
                    "[STAGING_QUARANTINE_EXIT] alloc_id=%s delay_s=%.3f "
                    "reason=%s quarantine_count=%s",
                    alloc_id,
                    delay_s,
                    reason,
                    self.staging_allocator.quarantine_count,
                )
                if self._fatal_shutdown is None:
                    os.kill(os.getppid(), signal.SIGQUIT)
                else:
                    self._fatal_shutdown(
                        -1,
                        self.staging_allocator.quarantine_count,
                        reason="staging-quarantine",
                    )

            timer = threading.Timer(delay_s, request_exit)
            timer.daemon = True
            self._quarantine_exit_timer = timer
            timer.start()
        logger.error(
            "[STAGING_QUARANTINE_EXIT] scheduled process-level exit in %.3fs "
            "(reason=%s)",
            delay_s,
            reason,
        )

    def _outstanding_alloc_ids(self, decode_req, receiver) -> set[int]:
        alloc_ids = {
            info[0]
            for info in getattr(receiver, "chunk_staging_infos", [])
            if info[0] >= 0
        }
        for item in getattr(decode_req, "_chunk_events", []):
            alloc_ids.add(item[1])
        return {
            alloc_id
            for alloc_id in alloc_ids
            if alloc_id in self.staging_allocator.allocations
        }

    def _terminalize_room(self, room: int, reason: str) -> set[int]:
        """Close the scatter/allocation fence, then quarantine live allocations."""
        lifecycle = self._room_lifecycles.get(room)
        decode_req = self._room_to_decode_req.get(room)
        receiver = self._room_to_receiver.get(room)
        if lifecycle is None or decode_req is None:
            return set()
        self.drop_pending_allocs(room)
        with lifecycle.lock:
            lifecycle.terminal = True
            alloc_ids = self._outstanding_alloc_ids(decode_req, receiver)
            if alloc_ids:
                lifecycle.quarantined = True
                for chunk_idx in lifecycle.chunk_states:
                    lifecycle.chunk_states[chunk_idx] = "TERMINAL"
        for alloc_id in alloc_ids:
            self.staging_allocator.quarantine(alloc_id, reason)
        return alloc_ids

    def _drain_scatter_bounded(self, room: int) -> None:
        """Wait briefly for claimed scatter without an unbounded synchronize."""
        stream = self.staging_allocator._scatter_stream
        if stream is None:
            return
        deadline = time.monotonic() + STAGING_SCATTER_DRAIN_TIMEOUT_S
        while True:
            try:
                if stream.query():
                    return
            except Exception:
                logger.exception(
                    "[STAGING_FATAL_EXIT] room=%s scatter query failed", room
                )
                if self._fatal_shutdown is None:
                    os.kill(os.getppid(), signal.SIGQUIT)
                else:
                    self._fatal_shutdown(room, 1, reason="staging-scatter-query-failed")
                raise
            if time.monotonic() >= deadline:
                reason = "staging-scatter-drain-timeout"
                logger.critical(
                    "[STAGING_FATAL_EXIT] room=%s scatter did not drain in %.3fs",
                    room,
                    STAGING_SCATTER_DRAIN_TIMEOUT_S,
                )
                if self._fatal_shutdown is None:
                    os.kill(os.getppid(), signal.SIGQUIT)
                else:
                    self._fatal_shutdown(room, 1, reason=reason)
                raise RuntimeError(f"[STAGING_SCATTER_DRAIN_TIMEOUT] room={room}")
            time.sleep(0.001)

    def fence_failed_room(self, room: int, reason: str) -> None:
        """Fence late scatter before the scheduler releases request KV pages."""
        alloc_ids = self._terminalize_room(room, reason)
        if not alloc_ids:
            return
        # A stuck CUDA drain escalates to fail-stop on its own (see
        # _drain_scatter_bounded); quarantine itself only isolates unless the
        # reason or the lost capacity says otherwise (see _on_quarantine).
        self._drain_scatter_bounded(room)

    def fail_staging_room(self, room: int, reason: str) -> None:
        """Fail closed after a staging invariant violation."""
        self.fence_failed_room(room, reason)
        from sglang.srt.disaggregation.base.conn import KVPoll

        self.kv_manager.record_failure(room, reason)
        self.kv_manager.update_status(room, KVPoll.Failed)
        receiver = self._room_to_receiver.get(room)
        if receiver is not None:
            receiver.conclude_state = KVPoll.Failed

    def register_decode_req(self, room: int, decode_req: DecodeRequest) -> None:
        # Completion is resource-driven (same model as upstream #30545): the
        # room is done once every rank reported Success, every scatter event
        # fired, and no staging allocation is still live. No chunk is special.
        decode_req._staging_scatter_done = False
        decode_req._staging_all_success = False
        decode_req._staging_success_ts = 0.0
        decode_req._chunk_events = []
        # Stall clock: None means "no allocations held / progress just made".
        # It starts counting only after the first accepted writer notification,
        # and is reset by every later progress event (chunk arrival, scatter
        # submit, event completion). Progress is
        # recorded by the decode thread while timeout checks run on the
        # scheduler thread, so serialize the timestamp and failure decision.
        decode_req._staging_progress_lock = threading.Lock()
        decode_req._staging_data_started = False
        decode_req._staging_stall_since = None
        decode_req._staging_stall_failed = False
        self._room_lifecycles[room] = StagingRoomLifecycle()
        self._room_to_decode_req[room] = decode_req
        self._room_to_receiver[room] = decode_req.kv_receiver

    def unregister_decode_req(self, room: int) -> None:
        # The lifecycle terminal flag closes new work before the maps disappear.
        # This lets release_room still discover and quarantine every live alloc.
        self.drop_pending_allocs(room)
        decode_req = self._room_to_decode_req.get(room)
        receiver = self._room_to_receiver.get(room)
        if decode_req is not None:
            self.release_room(room, decode_req, receiver)
        self._room_to_decode_req.pop(room, None)
        self._room_to_receiver.pop(room, None)
        self._room_lifecycles.pop(room, None)
        self.kv_manager._staging_ctx.room_receivers.pop(room, None)
        self.kv_manager._staging_ctx.room_bootstrap.pop(room, None)
        if hasattr(self.kv_manager, "_chunk_writer_counts"):
            self.kv_manager._chunk_writer_counts.pop(room, None)

    def release_room(self, room: int, decode_req: DecodeRequest, receiver) -> None:
        """Release allocations left by a failed or aborted staging room."""
        # Any allocation left at unregister has an outstanding remote grant.
        # Quarantine it before the bounded scatter drain; the independent exit
        # timer is armed before any waiting starts.
        quarantined = self._terminalize_room(room, "room-release")
        if quarantined:
            self._drain_scatter_bounded(room)

        chunk_infos = (
            getattr(receiver, "chunk_staging_infos", []) if receiver is not None else []
        )
        for chunk_idx, info in enumerate(chunk_infos):
            if info[0] >= 0:
                logger.error(
                    "[STAGING_RELEASE] room=%s chunk=%s alloc_id=%s "
                    "reason=quarantined",
                    room,
                    chunk_idx,
                    info[0],
                )
                chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        decode_req._chunk_events.clear()

    # ------------------------------------------------------------------
    # Scatter submission: called from decode_thread (background)
    # ------------------------------------------------------------------

    def submit_chunk_scatter(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
    ) -> bool:
        """Submit scatter after every writer covers the complete chunk.

        Called from decode_thread. Every chunk, including the request's final
        one, takes this path and records its CUDA event in ``_chunk_events``;
        the main thread later checks completion and frees the allocation.
        """
        lifecycle = self._room_lifecycles.get(room)
        if lifecycle is None:
            logger.warning(
                "[STAGING] submit_chunk_scatter: room=%s not registered, "
                "chunk_idx=%s. This should not happen if register_decode_req "
                "is called at kv_receiver.init() time.",
                room,
                chunk_idx,
            )
            return False
        violation = None
        with lifecycle.lock:
            if lifecycle.terminal:
                count = self.invariant_counters.increment("scatter_after_terminal")
                logger.error(
                    "[STAGING_INVARIANT] scatter_after_terminal=%s room=%s " "chunk=%s",
                    count,
                    room,
                    chunk_idx,
                )
                return False
            state = lifecycle.chunk_states.get(chunk_idx)
            if state in ("SCATTER_SUBMITTED", "SCATTER_DONE"):
                return True
            expected_geometry = lifecycle.chunk_geometry.get(chunk_idx)
            if state != "WRITABLE" or expected_geometry != (
                page_start,
                num_pages,
            ):
                lifecycle.terminal = True
                violation = (
                    f"[STAGING_GEOMETRY] scatter mismatch room={room} "
                    f"chunk={chunk_idx} state={state} got={(page_start, num_pages)} "
                    f"expected={expected_geometry}"
                )
            else:
                decode_req = self._room_to_decode_req.get(room)
                receiver = self._room_to_receiver.get(room)
                chunk_infos = (
                    receiver.chunk_staging_infos if receiver is not None else []
                )
                if decode_req is None or chunk_idx >= len(chunk_infos):
                    lifecycle.terminal = True
                    violation = (
                        f"[STAGING_GEOMETRY] missing allocation room={room} "
                        f"chunk={chunk_idx}"
                    )
                else:
                    alloc_id, staging_offset, _, _, allocated_pages = chunk_infos[
                        chunk_idx
                    ]
                    if (
                        staging_offset < 0
                        or alloc_id < 0
                        or allocated_pages != num_pages
                    ):
                        lifecycle.terminal = True
                        violation = (
                            f"[STAGING_GEOMETRY] invalid allocation room={room} "
                            f"chunk={chunk_idx} alloc_id={alloc_id} "
                            f"offset={staging_offset} pages={allocated_pages} "
                            f"expected_pages={num_pages}"
                        )
                    else:
                        # Claim and enqueue under the same short lifecycle lock.
                        # A terminal transition can therefore only observe work
                        # that is already present on scatter_stream.
                        lifecycle.chunk_states[chunk_idx] = "SCATTER_SUBMITTED"
                        self._note_staging_progress(decode_req)
                        ok = self._scatter_region(
                            staging_offset, page_start, num_pages, decode_req
                        )
                        event = torch.cuda.Event()
                        event.record(self.staging_allocator._scatter_stream)
                        decode_req._chunk_events.append((event, alloc_id, chunk_idx))
                        return ok
        logger.error(violation)
        self.fail_staging_room(room, violation)
        return False

    def is_staging_room(self, room: int) -> bool:
        """Check if a room is registered for staging scatter."""
        return room in self._room_to_decode_req

    def handle_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        engine_rank,
        peer_name="",
        chunk_writer_counts: Optional[dict] = None,
    ) -> Tuple[bool, bool]:
        """Validate and accumulate one writer's covered page interval."""
        legacy_submit = chunk_writer_counts is None
        if legacy_submit:
            # Mooncake's existing six-argument call passes its writer table in
            # peer_name and a session identity in engine_rank. Keep that local
            # API compatible without changing its wire format or source file.
            chunk_writer_counts = peer_name
            peer_name = str(engine_rank)
        lifecycle = self._room_lifecycles.get(room)
        decode_req = self._room_to_decode_req.get(room)
        if lifecycle is None or decode_req is None:
            logger.warning(
                "Staging chunk arrived for unregistered room=%s chunk=%d, skipping",
                room,
                chunk_idx,
            )
            return (False, False)
        violation = None
        accepted = False
        coverage_complete = False
        expected_geometry = None
        with lifecycle.lock:
            if lifecycle.terminal:
                count = self.invariant_counters.increment("scatter_after_terminal")
                logger.error(
                    "[STAGING_INVARIANT] scatter_after_terminal=%s room=%s "
                    "chunk=%s source=notification",
                    count,
                    room,
                    chunk_idx,
                )
                return (False, False)
            state = lifecycle.chunk_states.get(chunk_idx)
            expected_geometry = lifecycle.chunk_geometry.get(chunk_idx)
            if expected_geometry is None:
                lifecycle.terminal = True
                violation = (
                    f"[STAGING_GEOMETRY] notification without allocation "
                    f"room={room} chunk={chunk_idx} state={state}"
                )
            else:
                expected_start, expected_pages = expected_geometry
                expected_end = expected_start + expected_pages
                page_end = page_start + num_pages
                if (
                    num_pages <= 0
                    or page_start < expected_start
                    or page_end > expected_end
                ):
                    lifecycle.terminal = True
                    violation = (
                        f"[STAGING_GEOMETRY] notification out of bounds "
                        f"room={room} chunk={chunk_idx} state={state} "
                        f"got=[{page_start},{page_end}) "
                        f"expected=[{expected_start},{expected_end})"
                    )

            if violation is None:
                num_writers = self.num_writers_for(decode_req)
                if legacy_submit:
                    writer_slot = peer_name
                    writer_valid = True
                else:
                    prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
                    writer_slot = (engine_rank % prefill_tp) % num_writers
                    writer_valid = writer_slot in range(num_writers)
                if not writer_valid:
                    lifecycle.terminal = True
                    violation = (
                        f"[STAGING_WRITER] invalid slot room={room} chunk={chunk_idx} "
                        f"engine_rank={engine_rank} slot={writer_slot} "
                        f"expected=0..{num_writers - 1}"
                    )
                else:
                    interval = (page_start, page_end)
                    chunk_intervals = lifecycle.writer_intervals.setdefault(
                        chunk_idx, {}
                    )
                    writer_intervals = chunk_intervals.setdefault(writer_slot, set())
                    if interval in writer_intervals:
                        count = self.invariant_counters.increment("duplicate_writer")
                        logger.warning(
                            "[STAGING_DUPLICATE_WRITER] duplicate_writer=%s "
                            "room=%s chunk=%s "
                            "engine_rank=%s writer_slot=%s interval=%s peer=%s",
                            count,
                            room,
                            chunk_idx,
                            engine_rank,
                            writer_slot,
                            interval,
                            peer_name,
                        )
                        return (False, False)
                    overlap = next(
                        (
                            existing
                            for existing in writer_intervals
                            if max(existing[0], page_start) < min(existing[1], page_end)
                        ),
                        None,
                    )
                    if overlap is not None:
                        lifecycle.terminal = True
                        violation = (
                            f"[STAGING_GEOMETRY] overlapping writer interval "
                            f"room={room} chunk={chunk_idx} writer_slot={writer_slot} "
                            f"got={interval} existing={overlap}"
                        )
                    elif state != "WRITABLE":
                        lifecycle.terminal = True
                        violation = (
                            f"[STAGING_LIFECYCLE] new writer after scatter "
                            f"room={room} chunk={chunk_idx} state={state} "
                            f"writer_slot={writer_slot}"
                        )
                    else:
                        writer_intervals.add(interval)
                        seen = lifecycle.seen_writer_slots.setdefault(chunk_idx, set())
                        seen.add(writer_slot)
                        writer_counts = chunk_writer_counts[room][chunk_idx]
                        coverage_key = (writer_slot, page_start, page_end)
                        if hasattr(writer_counts, "add"):
                            writer_counts.add(coverage_key)
                        else:
                            writer_counts.append(coverage_key)
                        self._note_staging_data_started(decode_req)
                        coverage_complete = self._chunk_coverage_complete_locked(
                            lifecycle, chunk_idx, num_writers
                        )
                        accepted = True
                        if not legacy_submit:
                            return (accepted, coverage_complete)
        if violation is None and legacy_submit:
            if coverage_complete:
                self.submit_chunk_scatter(room, chunk_idx, *expected_geometry)
            return (accepted, coverage_complete)
        logger.error(violation)
        self.fail_staging_room(room, violation)
        return (False, False)

    def submit_last_scatter_async(self, room: int) -> bool:
        """Record all-ranks Success for a room.

        Scatter is fully arrival-driven: every chunk, including the request's
        final one, is scattered by handle_chunk_arrived once its writer
        coverage is complete. This only marks that the transport layer has
        delivered everything; advance_scatter completes the room once no
        scatter event is pending and no staging allocation is still live.
        Deciding completion from "which notification was tagged last" used
        to leave rooms parked forever when chunked prefill put two logical
        chunks into one staging chunk (2026-09-03 C512 reproduction).
        """
        lifecycle = self._room_lifecycles.get(room)
        decode_req = self._room_to_decode_req.get(room)
        if lifecycle is None or decode_req is None:
            logger.warning(
                "[STAGING] submit_last_scatter_async: room=%s not registered. "
                "This should not happen if register_decode_req is called at "
                "kv_receiver.init() time.",
                room,
            )
            return False
        with lifecycle.lock:
            if lifecycle.terminal:
                return False
            if not decode_req._staging_all_success:
                # Timestamp before flag so a concurrent reader never sees a
                # set flag with a zero timestamp.
                decode_req._staging_success_ts = time.monotonic()
                decode_req._staging_all_success = True
        return True

    def release_unwritten_chunks(self, room: int) -> None:
        """Free allocations of chunks that will never receive data.

        Used when every prefill rank reported ``aux_nokv`` (decode-side radix
        cache hit): STAGING_REQ already allocated ring space for the room, but
        no writer will ever cover it, so release it and mark the chunks
        terminal so a late writer is rejected instead of scattered.
        """
        lifecycle = self._room_lifecycles.get(room)
        decode_req = self._room_to_decode_req.get(room)
        receiver = self._room_to_receiver.get(room)
        if lifecycle is None or decode_req is None or receiver is None:
            return
        freed = False
        with lifecycle.lock:
            if lifecycle.terminal:
                return
            chunk_infos = getattr(receiver, "chunk_staging_infos", [])
            for chunk_idx, info in enumerate(chunk_infos):
                alloc_id = info[0]
                if alloc_id < 0 or lifecycle.writer_intervals.get(chunk_idx):
                    continue
                if lifecycle.chunk_states.get(chunk_idx, "WRITABLE") != "WRITABLE":
                    continue
                lifecycle.chunk_states[chunk_idx] = "SCATTER_DONE"
                chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
                freed |= self._free_allocation(alloc_id, decode_req)
        if freed:
            self._service_pending_allocs()

    # ------------------------------------------------------------------
    # Event check + free: called from main thread (pop_transferred)
    # ------------------------------------------------------------------

    def is_done(self, decode_req: DecodeRequest) -> bool:
        """Return True if staging scatter is complete for this request."""
        return decode_req._staging_scatter_done and not decode_req._chunk_events

    def advance_scatter(self, decode_req: DecodeRequest) -> None:
        """Check CUDA events and free completed staging allocations.

        Scatter kernels have already been submitted by the decode_thread
        (via submit_chunk_scatter / submit_last_scatter_async).  This
        method only polls the recorded events and releases staging memory.
        """
        room = decode_req.req.bootstrap_room
        lifecycle = self._room_lifecycles.get(room)
        if lifecycle is None:
            return
        chunk_events = decode_req._chunk_events
        freed = False
        if chunk_events:
            for i in range(len(chunk_events) - 1, -1, -1):
                event, alloc_id, chunk_idx = chunk_events[i]
                if event.query():
                    with lifecycle.lock:
                        self._note_staging_progress(decode_req)
                        chunk_events.pop(i)
                        lifecycle.chunk_states[chunk_idx] = "SCATTER_DONE"
                        receiver = self._room_to_receiver.get(room)
                        chunk_infos = (
                            getattr(receiver, "chunk_staging_infos", [])
                            if receiver is not None
                            else []
                        )
                        if (
                            chunk_idx < len(chunk_infos)
                            and chunk_infos[chunk_idx][0] == alloc_id
                        ):
                            chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
                        if not lifecycle.terminal:
                            freed |= self._free_allocation(alloc_id, decode_req)
        if freed:
            # Outside lifecycle.lock: the queue head may be this very room.
            self._service_pending_allocs()

        # Resource-driven completion (same model as upstream #30545): all
        # ranks reported Success, every scatter event fired, and no staging
        # allocation is still live. A chunk whose data is still in flight
        # keeps info[0] >= 0 and therefore keeps the room open.
        if (
            decode_req._staging_all_success
            and not decode_req._staging_scatter_done
            and not chunk_events
        ):
            receiver = self._room_to_receiver.get(room)
            chunk_infos = (
                getattr(receiver, "chunk_staging_infos", [])
                if receiver is not None
                else []
            )
            if not any(info[0] >= 0 for info in chunk_infos):
                decode_req._staging_scatter_done = True

        # Consume completed events before checking the watchdog. Otherwise a
        # scatter event that completes on the timeout boundary can be
        # mistaken for a stalled allocation and fail an already-finished room.
        self._check_room_stall(room, decode_req)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: DecodeRequest,
    ) -> bool:
        """Submit scatter kernels for a staging region to scatter_stream.

        May be called from the decode_thread (background).  All GPU work
        runs on scatter_stream so that the decode_thread never blocks on
        the default stream (which carries the main-thread forward pass).
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            scatter_staging_to_kv,
        )

        k_buffers = self.kv_buffer_info["k_buffers"]
        v_buffers = self.kv_buffer_info["v_buffers"]
        page_size = self.kv_buffer_info["page_size"]
        dst_tp_rank = self.kv_manager.kv_args.engine_rank % self.decode_tp

        device = k_buffers[0].device
        torch.cuda.set_device(device)

        if self.staging_allocator._scatter_stream is None:
            self.staging_allocator._scatter_stream = torch.cuda.Stream(device=device)

        scatter_stream = self.staging_allocator._scatter_stream

        staging_view = self.staging_allocator.buffer.buffer[staging_offset:]

        req_pool_idx = decode_req.req.req_pool_idx
        token_start = page_start * page_size
        token_end = token_start + num_pages * page_size
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size

        with torch.cuda.stream(scatter_stream):
            kv_indices = self.scheduler.req_to_token_pool.req_to_token[
                req_pool_idx, token_start:token_end
            ]
            if page_size > 1:
                page_idx_tensor = kv_indices[::page_size] // page_size
            else:
                page_idx_tensor = kv_indices

            scatter_staging_to_kv(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                self.decode_tp,
                dst_tp_rank,
                self.total_kv_heads,
            )

        return True

    def _room_holds_allocations(self, decode_req: DecodeRequest, receiver) -> bool:
        """True if this room currently pins any staging ring allocation."""
        if decode_req._chunk_events:
            return True
        chunk_infos = (
            getattr(receiver, "chunk_staging_infos", []) if receiver is not None else []
        )
        return any(info[0] >= 0 for info in chunk_infos)

    def completion_snapshot(
        self, room: int, decode_req, receiver, metadata_room
    ) -> str:
        """One-line state dump used to attribute a completion timeout."""
        lifecycle = self._room_lifecycles.get(room)
        status = getattr(self.kv_manager, "transfer_statuses", {}).get(room)
        received = getattr(status, "received_kvs_per_pp", None) or {}
        success_ts = getattr(decode_req, "_staging_success_ts", 0.0) or 0.0
        fields = {
            "scatter_done": getattr(decode_req, "_staging_scatter_done", None),
            "all_success": getattr(decode_req, "_staging_all_success", None),
            "success_age_s": (
                round(time.monotonic() - success_ts, 1) if success_ts else None
            ),
            "chunk_events": len(getattr(decode_req, "_chunk_events", None) or []),
            "data_started": getattr(decode_req, "_staging_data_started", None),
            "lifecycle": (
                None
                if lifecycle is None
                else {
                    "terminal": lifecycle.terminal,
                    "quarantined": lifecycle.quarantined,
                    "chunk_states": dict(lifecycle.chunk_states),
                    "geometry": dict(lifecycle.chunk_geometry),
                }
            ),
            "chunk_infos": list(getattr(receiver, "chunk_staging_infos", None) or []),
            "require_staging": getattr(receiver, "require_staging", None),
            "received_aux": getattr(status, "received_aux", None),
            "expected_kvs": dict(getattr(status, "expected_kvs_per_pp", None) or {}),
            "received_kvs": {k: sorted(v) for k, v in received.items()},
            "metadata_index": getattr(decode_req, "metadata_buffer_index", None),
            "metadata_room": metadata_room,
        }
        return " ".join(f"{k}={v}" for k, v in fields.items())

    def _note_staging_progress(self, decode_req: DecodeRequest) -> None:
        """Atomically refresh a room's stall clock after real progress."""
        with decode_req._staging_progress_lock:
            # Once the scheduler has committed the failure decision, late
            # notifications must not revive the room.
            if not decode_req._staging_stall_failed:
                decode_req._staging_stall_since = None

    def _note_staging_data_started(self, decode_req: DecodeRequest) -> None:
        """Mark the first accepted writer and refresh the stall clock atomically."""
        with decode_req._staging_progress_lock:
            if not decode_req._staging_stall_failed:
                decode_req._staging_data_started = True
                decode_req._staging_stall_since = None

    def _check_room_stall(self, room: int, decode_req: DecodeRequest) -> None:
        """Fail a room that holds ring allocations but makes no progress.

        Runs on the scheduler main thread from advance_scatter. Failing the
        room through the receiver status (same mechanism as
        _check_waiting_timeout) routes it into the existing
        Failed -> pop_transferred -> unregister_decode_req -> release_room
        path, which frees its allocations and unpins the shared ring.
        The clock only runs after the first accepted writer and while allocations
        are held, so allocation-only rooms waiting for data are never failed here.
        """
        receiver = self._room_to_receiver.get(room)
        now = time.monotonic()
        with decode_req._staging_progress_lock:
            if decode_req._staging_stall_failed:
                return
            if not decode_req._staging_data_started:
                decode_req._staging_stall_since = None
                return
            if not self._room_holds_allocations(decode_req, receiver):
                decode_req._staging_stall_since = None
                return
            since = decode_req._staging_stall_since
            if since is None:
                decode_req._staging_stall_since = now
                return
            elapsed = now - since
            if elapsed <= STAGING_ROOM_STALL_TIMEOUT_S:
                return
            # Commit under the same lock used by progress writers. This closes
            # the stale-read race where the decode thread refreshed the clock
            # after the scheduler read it but before it marked the room failed.
            decode_req._staging_stall_failed = True
        self._terminalize_room(room, "staging-stall")
        watermark = self.staging_allocator.get_watermark()
        logger.error(
            "[STAGING_STALL] room=%s held staging allocations with no "
            "progress for %.0fs (watermark=%s pending_events=%d); fencing "
            "and quarantining the room until process restart.",
            room,
            elapsed,
            watermark,
            len(decode_req._chunk_events),
        )
        from sglang.srt.disaggregation.base.conn import KVPoll

        self.kv_manager.record_failure(
            room,
            f"[STAGING_STALL] no staging scatter progress for {elapsed:.0f}s; "
            f"failed to protect the shared staging ring",
        )
        self.kv_manager.update_status(room, KVPoll.Failed)
        # A receiver that already concluded Success caches conclude_state and
        # never re-reads request_status (its scatter can still stall after
        # that -- the exact 2026-08-20 incident state). Force the cache so the
        # next poll returns Failed and pop_transferred routes the room into
        # unregister_decode_req -> release_room.
        if receiver is not None:
            receiver.conclude_state = KVPoll.Failed

    def _free_allocation(self, alloc_id: int, decode_req: DecodeRequest) -> bool:
        """Release an extent. Returns True if it was freed.

        Callers hold a room's lifecycle lock here; servicing the pending queue
        takes the head room's lifecycle lock, so it must run after the caller
        drops its own lock (see advance_scatter / release_unwritten_chunks).
        """
        return self.staging_allocator.free(alloc_id)

    def release_allocation(self, alloc_id: int, decode_req: DecodeRequest) -> bool:
        """Free an extent and service the pending queue. Only for callers that
        hold no lifecycle lock."""
        freed = self._free_allocation(alloc_id, decode_req)
        if freed:
            self._service_pending_allocs()
        return freed


def is_watermark_ready(
    staging_state, session_id: str, alloc_round: int, alloc_end: int
) -> bool:
    """Non-blocking check: is the staging region safe to write?"""
    if alloc_round <= 0:
        return True
    prev_round = alloc_round - 1
    wm_round, wm_tail = staging_state.remote_watermarks.get(session_id, (0, 0))
    return prev_round < wm_round or (prev_round == wm_round and alloc_end <= wm_tail)


def handle_watermark_msg(staging_ctx, msg_parts) -> None:
    """Process a WATERMARK message and update remote watermark tracking."""
    wm_round = int(msg_parts[1].decode("ascii"))
    wm_tail = int(msg_parts[2].decode("ascii"))
    wm_session = msg_parts[3].decode("ascii") if len(msg_parts) > 3 else ""
    with staging_ctx.watermark_cv:
        prev = staging_ctx.remote_watermarks.get(wm_session, (0, 0))
        if (wm_round, wm_tail) > prev:
            staging_ctx.remote_watermarks[wm_session] = (
                wm_round,
                wm_tail,
            )
        staging_ctx.watermark_cv.notify_all()


def handle_staging_rsp(msg_parts, transfer_infos: dict) -> None:
    """Process a STAGING_RSP message and update transfer info with allocation."""
    stg_room = int(msg_parts[1].decode("ascii"))
    stg_chunk_idx = int(msg_parts[2].decode("ascii"))
    stg_offset = int(msg_parts[3].decode("ascii"))
    stg_round = int(msg_parts[4].decode("ascii"))
    stg_end = int(msg_parts[5].decode("ascii"))
    stg_session = msg_parts[6].decode("ascii")
    room_infos = transfer_infos.get(stg_room, {})
    tinfo = room_infos.get(stg_session)
    if tinfo is not None:
        if tinfo.staging is None:
            tinfo.staging = StagingTransferInfo()
        tinfo.staging.set_chunk(stg_chunk_idx, stg_offset, stg_round, stg_end)
    else:
        logger.warning(
            "STAGING_RSP RECV but tinfo=None room=%s chunk=%d session=%s",
            stg_room,
            stg_chunk_idx,
            stg_session,
        )


# ======================================================================
# Staging data structures and protocol utilities
# ======================================================================


@dataclasses.dataclass
class StagingTransferInfo:
    """Per-chunk staging allocation info attached to a TransferInfo."""

    offsets: List[int] = dataclasses.field(default_factory=lambda: [-1])
    rounds: List[int] = dataclasses.field(default_factory=lambda: [0])
    ends: List[int] = dataclasses.field(default_factory=lambda: [-1])

    def set_chunk(self, idx: int, offset: int, rnd: int, end: int):
        while len(self.offsets) <= idx:
            self.offsets.append(-1)
            self.rounds.append(0)
            self.ends.append(-1)
        self.offsets[idx] = offset
        self.rounds[idx] = rnd
        self.ends[idx] = end


@dataclasses.dataclass
class StagingRegisterInfo:
    """Staging buffer registration info attached to a KVArgsRegisterInfo."""

    base_ptr: int = 0
    total_size: int = 0

    @classmethod
    def from_zmq_fields(
        cls, msg: list, msg_start_offset: int
    ) -> Optional[StagingRegisterInfo]:
        i = msg_start_offset
        base_ptr = (
            struct.unpack("Q", msg[i])[0] if len(msg) > i and len(msg[i]) == 8 else 0
        )
        total_size = (
            int(msg[i + 1].decode("ascii"))
            if len(msg) > i + 1 and len(msg[i + 1]) > 0
            else 0
        )
        if base_ptr == 0 and total_size == 0:
            return None
        return cls(base_ptr=base_ptr, total_size=total_size)


class PrefillStagingStrategy:
    """Prefill-side staging transfer: readiness check + gather-RDMA execution.

    Encapsulates the decision logic (chunk index calculation, staging offset
    lookup, watermark readiness) and delegates actual RDMA to the kv_manager.
    """

    def __init__(self, kv_manager, staging_buffer):
        self.kv_manager = kv_manager
        self.staging_buffer = staging_buffer
        page_size = kv_manager.kv_buffer_tensors["page_size"]
        cps = kv_manager.server_args.chunked_prefill_size or 8192
        self.full_chunk_pages = max(1, cps // page_size)

    def check_ready(
        self,
        req,
        kv_chunk_index_start: int,
        num_chunk_pages: int,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, int, int, int, int]:
        """Check if staging offset and watermark are ready for this chunk.

        Args:
            req: transfer request with a ``.staging`` attribute.
            kv_chunk_index_start: page-level start index for this chunk.
            num_chunk_pages: number of pages in this chunk.
            session_id: identifier used for watermark lookup. Falls back to
                ``req.mooncake_session_id`` when *None* (mooncake compat).

        Returns (ready, chunk_idx, offset, round, end).
        offset == ALLOC_OVERSIZED means permanent failure (fall back to slice).
        offset == -1 means allocation pending (re-enqueue).
        """
        from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

        chunk_idx = (
            kv_chunk_index_start // self.full_chunk_pages
            if self.full_chunk_pages > 0
            else 0
        )

        stg = req.staging
        if stg is None or chunk_idx >= len(stg.offsets):
            return (False, chunk_idx, -1, 0, -1)

        c_offset = stg.offsets[chunk_idx]
        if c_offset == StagingAllocator.ALLOC_OVERSIZED:
            return (False, chunk_idx, StagingAllocator.ALLOC_OVERSIZED, 0, -1)
        if c_offset < 0:
            return (False, chunk_idx, -1, 0, -1)

        c_round = stg.rounds[chunk_idx]
        c_end = stg.ends[chunk_idx]

        if session_id is None:
            session_id = req.mooncake_session_id
        if not self.kv_manager._is_watermark_ready(session_id, c_round, c_end):
            return (False, chunk_idx, c_offset, c_round, c_end)

        return (True, chunk_idx, c_offset, c_round, c_end)

    def transfer(
        self,
        session_id: str,
        prefill_kv_indices,
        dst_staging_ptr: int,
        dst_staging_size: int,
        target_info,
    ) -> int:
        """Execute staged transfer (gather + RDMA).

        Returns 0 on success, -1 to signal fallback to slice path.
        """
        try:
            return self.kv_manager.send_kvcache_staged(
                session_id,
                prefill_kv_indices,
                dst_staging_ptr,
                dst_staging_size,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                staging_buffer=self.staging_buffer,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Staging] KV transfer via staging buffer failed: {e}. "
                f"session={session_id}"
            ) from e


def _get_custom_mem_pool(device: str):
    """Get custom memory pool for staging buffer allocation (backend-agnostic).

    Returns (custom_mem_pool, pool_type) tuple. custom_mem_pool may be None
    if no custom pool is configured.
    """
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )

    _, custom_mem_pool, pool_type = init_mooncake_custom_mem_pool(device)
    if custom_mem_pool is None:
        logger.info(
            "Staging buffer using cudaMalloc (no custom mem pool). "
            "This works for all GPU architectures. "
            "For NVLink/MNNVL transport, set SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )
    return custom_mem_pool, pool_type


def init_staging_buffers(register_fn, kv_args, count: int) -> list:
    """Create prefill-side staging buffers and register them with the transport.

    Args:
        register_fn: callable(ptr: int, size: int) that registers a memory
            region with the transport backend.
        kv_args: KVArgs with gpu_id.
        count: number of staging buffers to create.

    Returns list of StagingBuffer instances.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
    from sglang.srt.environ import envs

    size_mb = envs.SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB.get()
    size_bytes = size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    return buffers


def init_staging_allocator(register_fn, kv_args):
    """Create decode-side staging ring-buffer allocator and register with transport.

    Args:
        register_fn: callable(ptr: int, size: int) that registers a memory
            region with the transport backend.
        kv_args: KVArgs with gpu_id.

    Returns a StagingAllocator instance.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.environ import envs

    pool_size_mb = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get()
    pool_size_bytes = pool_size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)
    allocator = StagingAllocator(pool_size_bytes, device, gpu_id, custom_mem_pool)
    register_fn(allocator.get_base_ptr(), allocator.get_total_size())
    return allocator


def handle_staging_req(
    msg,
    staging_allocator,
    kv_args,
    attn_tp_size: int,
    prefill_attn_tp_size: int,
    kv_buffer_tensors,
    room_receivers: dict,
    room_bootstrap: dict,
    staging_handler: Optional[DecodeStagingHandler] = None,
    full_chunk_pages: Optional[int] = None,
):
    """Allocate staging for a chunk on-demand and send STAGING_RSP to prefill.

    Deduplicates: multiple prefill TP ranks requesting the same (room, chunk_idx)
    only allocate once.  Sends ALLOC_OVERSIZED on permanent failure.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

    room = int(msg[1].decode("ascii"))
    chunk_idx = int(msg[2].decode("ascii"))
    chunk_num_pages = int(msg[3].decode("ascii"))
    session_id = msg[4].decode("ascii")

    if staging_allocator is None:
        logger.warning(
            "STAGING_REQ ignored: allocator is None room=%s chunk=%s",
            room,
            chunk_idx,
        )
        return

    receiver = room_receivers.get(room)
    if receiver is None:
        logger.warning(
            "STAGING_REQ dropped: no receiver for room=%s chunk=%s session=%s",
            room,
            chunk_idx,
            session_id,
        )
        return
    infos = getattr(receiver, "chunk_staging_infos", [])
    if staging_handler is None:
        staging_handler = getattr(staging_allocator, "_staging_handler", None)
    if staging_handler is None:
        logger.error(
            "[STAGING_REQ] dropped without handler room=%s chunk=%s", room, chunk_idx
        )
        return
    if full_chunk_pages is None:
        cps = staging_handler.scheduler.server_args.chunked_prefill_size or 8192
        full_chunk_pages = max(1, cps // kv_args.page_size)
    lifecycle = staging_handler._room_lifecycles.get(room)
    if lifecycle is None:
        logger.warning(
            "[STAGING_REQ] dropped without lifecycle room=%s chunk=%s",
            room,
            chunk_idx,
        )
        return

    from sglang.srt.disaggregation.common.staging_buffer import (
        compute_staging_layout,
        resolve_total_kv_heads,
    )

    page_size = kv_args.page_size
    kv_item_lens = kv_args.kv_item_lens
    num_kv_layers = len(kv_item_lens) // 2
    decode_bytes_per_token = kv_item_lens[0] // page_size
    total_kv_heads = resolve_total_kv_heads(kv_args, attn_tp_size)
    dst_heads_per_rank = max(1, total_kv_heads // max(1, attn_tp_size))
    bytes_per_head_per_token = decode_bytes_per_token // dst_heads_per_rank
    dst_tp_rank = kv_args.engine_rank % max(1, attn_tp_size)
    chunk_tokens = chunk_num_pages * page_size
    _, _, required = compute_staging_layout(
        prefill_attn_tp_size,
        attn_tp_size,
        dst_tp_rank,
        total_kv_heads,
        chunk_tokens,
        bytes_per_head_per_token,
        num_kv_layers,
    )

    expected_geometry = (chunk_idx * full_chunk_pages, chunk_num_pages)
    violation = None
    pending = False
    with lifecycle.lock:
        state = lifecycle.chunk_states.get(chunk_idx)
        if lifecycle.terminal or state in (
            "SCATTER_SUBMITTED",
            "SCATTER_DONE",
            "TERMINAL",
        ):
            count = staging_handler.invariant_counters.increment("alloc_after_terminal")
            logger.error(
                "[STAGING_INVARIANT] alloc_after_terminal=%s room=%s chunk=%s "
                "state=%s; dropping late STAGING_REQ",
                count,
                room,
                chunk_idx,
                state,
            )
            return
        previous_geometry = lifecycle.chunk_geometry.get(chunk_idx)
        if previous_geometry is not None and previous_geometry != expected_geometry:
            lifecycle.terminal = True
            violation = (
                f"[STAGING_GEOMETRY] STAGING_REQ mismatch room={room} "
                f"chunk={chunk_idx} got={expected_geometry} "
                f"expected={previous_geometry}"
            )
        else:
            lifecycle.chunk_geometry[chunk_idx] = expected_geometry

        if violation is None and chunk_idx < len(infos) and infos[chunk_idx][0] >= 0:
            _, offset, rnd, end, _ = infos[chunk_idx]
            lifecycle.chunk_states.setdefault(chunk_idx, "WRITABLE")
        elif (
            violation is None
            and chunk_idx < len(infos)
            and infos[chunk_idx][1] == StagingAllocator.ALLOC_OVERSIZED
        ):
            offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
            lifecycle.chunk_states.setdefault(chunk_idx, "OVERSIZED")
        elif violation is None and state == "PENDING_ALLOC":
            # Another prefill rank asked for the same chunk while it is still
            # queued for an extent; the grant will answer every rank at once.
            return
        elif violation is None:
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            attainable = staging_allocator.max_attainable_extent()
            if required > attainable:
                logger.error(
                    "[STAGING_REQ] chunk exceeds the staging pool room=%s chunk=%d "
                    "(need %d bytes, largest attainable extent=%d bytes, pool "
                    "total=%d bytes). Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB "
                    "or restart if quarantine has fragmented the pool.",
                    room,
                    chunk_idx,
                    required,
                    attainable,
                    staging_allocator.total_size,
                )
                offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
                infos[chunk_idx] = (
                    -1,
                    StagingAllocator.ALLOC_OVERSIZED,
                    0,
                    -1,
                    chunk_num_pages,
                )
                lifecycle.chunk_states[chunk_idx] = "OVERSIZED"
            else:
                result = staging_allocator.assign(required)
                if result is None:
                    # Pool is full: queue the request; a release grants it in
                    # FIFO order and answers with STAGING_RSP then.
                    infos[chunk_idx] = (-1, -1, 0, -1, chunk_num_pages)
                    lifecycle.chunk_states[chunk_idx] = "PENDING_ALLOC"
                    pending = True
                else:
                    alloc_id, offset, rnd = result
                    end = offset + required
                    infos[chunk_idx] = (alloc_id, offset, rnd, end, chunk_num_pages)
                    lifecycle.chunk_states[chunk_idx] = "WRITABLE"

    if violation is not None:
        logger.error(violation)
        staging_handler.fail_staging_room(room, violation)
        return

    if pending:
        staging_handler.enqueue_pending_alloc(
            room, chunk_idx, required, session_id, chunk_num_pages
        )
        # A release between the failed assign above and the enqueue would have
        # found an empty queue; service once more so that extent is not idle
        # until the next release.
        staging_handler._service_pending_allocs()
        return
    send_staging_rsp(
        receiver, room_bootstrap, room, chunk_idx, offset, rnd, end, session_id
    )


def send_staging_rsp(
    receiver,
    room_bootstrap: dict,
    room: int,
    chunk_idx: int,
    offset: int,
    rnd: int,
    end: int,
    session_id: str,
) -> None:
    """Answer a STAGING_REQ on every bootstrap endpoint of the room."""
    bootstrap_infos = room_bootstrap.get(room)
    if not bootstrap_infos:
        return
    for bi in bootstrap_infos:
        try:
            sock, lock = receiver._connect_to_bootstrap_server(bi)
            with lock:
                sock.send_multipart(
                    [
                        b"STAGING_RSP",
                        str(room).encode("ascii"),
                        str(chunk_idx).encode("ascii"),
                        str(offset).encode("ascii"),
                        str(rnd).encode("ascii"),
                        str(end).encode("ascii"),
                        session_id.encode("ascii"),
                    ]
                )
        except Exception:
            pass


def send_staging_fail(
    room: int, transfer_infos: dict, prefetch_sockets: dict, reason: str
) -> int:
    """Tell every decode peer of ``room`` that the prefill side failed it.

    Without this the decode only learns about a prefill transfer failure from
    its own waiting timeout (Dynamo consumes the prefill stream in the
    background and never cancels the decode request). Returns the number of
    peers notified. Best effort: a peer that cannot be reached is skipped.
    """
    import zmq
    from sglang.srt.utils.network import NetworkAddress

    notified = 0
    for tinfo in (transfer_infos.get(room) or {}).values():
        try:
            na = NetworkAddress(tinfo.endpoint, tinfo.dst_port)
            ep = na.to_tcp()
            if ep not in prefetch_sockets:
                sock = zmq.Context().socket(zmq.PUSH)
                if na.is_ipv6:
                    sock.setsockopt(zmq.IPV6, 1)
                sock.connect(ep)
                prefetch_sockets[ep] = sock
            prefetch_sockets[ep].send_multipart(
                [
                    b"STAGING_FAIL",
                    str(room).encode("ascii"),
                    reason.encode("utf-8", errors="replace")[:512],
                ]
            )
            notified += 1
        except Exception:
            logger.exception("[STAGING_FAIL] could not notify decode for room=%s", room)
    return notified


def prefetch_staging_reqs(
    room: int,
    transfer_infos: dict,
    kv_buffer_tensors: dict,
    chunked_prefill_size: int,
    staging_requested: set,
    prefetch_sockets: dict,
) -> None:
    """Send STAGING_REQ for all chunks before the prefill forward starts.

    Called from the scheduler right after batch formation, so that decode
    allocates staging during the GPU forward pass.
    """
    import zmq

    from sglang.srt.utils.network import NetworkAddress

    page_size = kv_buffer_tensors["page_size"]
    cps = chunked_prefill_size or 8192
    full_chunk_pages = max(1, cps // page_size)

    for session_id, tinfo in transfer_infos[room].items():
        # mooncake exposes is_dummy as a dataclass bool field, NIXL exposes it
        # as a method (it consults decode_prefix_len). Normalize via callable()
        # so this shared helper works for either backend; treating a bound
        # method as truthy (the previous behavior) silently dropped every
        # STAGING_REQ on NIXL and deadlocked the prefill transfer worker.
        is_dummy_attr = tinfo.is_dummy
        if is_dummy_attr() if callable(is_dummy_attr) else is_dummy_attr:
            continue
        total_pages = len(tinfo.dst_kv_indices)
        if total_pages == 0:
            continue
        num_chunks = (total_pages + full_chunk_pages - 1) // full_chunk_pages

        for chunk_idx in range(num_chunks):
            stg_key = (room, chunk_idx, session_id)
            if stg_key in staging_requested:
                continue
            staging_requested.add(stg_key)

            remaining = total_pages - chunk_idx * full_chunk_pages
            chunk_pages = min(full_chunk_pages, remaining)
            try:
                na = NetworkAddress(tinfo.endpoint, tinfo.dst_port)
                ep = na.to_tcp()
                if ep not in prefetch_sockets:
                    sock = zmq.Context().socket(zmq.PUSH)
                    if na.is_ipv6:
                        sock.setsockopt(zmq.IPV6, 1)
                    sock.connect(ep)
                    prefetch_sockets[ep] = sock
                prefetch_sockets[ep].send_multipart(
                    [
                        b"STAGING_REQ",
                        str(room).encode("ascii"),
                        str(chunk_idx).encode("ascii"),
                        str(chunk_pages).encode("ascii"),
                        session_id.encode("ascii"),
                    ]
                )
            except Exception:
                staging_requested.discard(stg_key)
