"""
Staging handler for heterogeneous TP KV cache transfer.

Isolates staging scatter lifecycle from decode.py and conn.py.
Generic (backend-agnostic) code is at the top; mooncake-specific
protocol code is at the bottom.
"""

from __future__ import annotations

import dataclasses
import logging
import os
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
STAGING_WATERMARK_SEND_RETRY_S = 0.001
STAGING_WATERMARK_SEND_WARN_AFTER_S = 5.0
STAGING_WATERMARK_SEND_WARN_EVERY_S = 30.0
STAGING_WATERMARK_SEND_ERROR_EVERY_S = 30.0

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
    ):
        self.kv_manager = kv_manager
        self.staging_allocator = staging_allocator
        self.kv_buffer_info = kv_buffer_info
        self.decode_tp = decode_tp
        self.total_kv_heads = total_kv_heads
        self.tp_rank = tp_rank
        self.scheduler = scheduler
        self._room_to_decode_req: dict = {}
        # Failure paths clear decode_req.kv_receiver before unregistering the
        # room, so retain the receiver needed to release staging allocations.
        self._room_to_receiver: dict = {}
        self._room_lifecycles: dict[int, StagingRoomLifecycle] = {}
        self.invariant_counters = staging_allocator.invariant_counters
        self.staging_allocator._staging_handler = self
        self._wm_subscribers: dict = {}
        self._wm_next_generation = 0
        self._wm_send_cv = threading.Condition()
        self._wm_send_pending: dict = {}
        self._wm_send_wait_started: dict = {}
        self._wm_send_last_log: dict = {}
        self._wm_send_error_last_log: dict = {}
        threading.Thread(
            target=self._watermark_sender_loop,
            daemon=True,
            name="staging-watermark-sender",
        ).start()

    def register_wm_subscriber(self, receiver, session_id: str) -> None:
        """Register or refresh a prefill watermark subscriber.

        A new session on the same endpoints bumps the generation. Sender work
        captured for the old generation is then discarded instead of being
        requeued onto the replacement receiver.
        """
        if receiver is None or not getattr(receiver, "bootstrap_infos", None):
            return
        key = tuple(str(bi) for bi in receiver.bootstrap_infos)
        with self._wm_send_cv:
            previous = self._wm_subscribers.get(key)
            if previous is None or previous[1] != session_id:
                self._wm_next_generation += 1
                generation = self._wm_next_generation
                self._wm_subscribers[key] = (receiver, session_id, generation)
                wm_round, wm_tail = self.staging_allocator.get_watermark()
                self._wm_send_pending[key] = (
                    receiver,
                    session_id,
                    wm_round,
                    wm_tail,
                    generation,
                )
                logger.info(
                    "STAGING_WATERMARK_SUBSCRIBER_REGISTER subscriber=%s "
                    "session=%s generation=%s replaced=%s",
                    key,
                    session_id,
                    generation,
                    previous is not None,
                )
            else:
                # A receiver is request-scoped in this integration. Refresh it
                # without bumping generation when the peer session is unchanged.
                generation = previous[2]
                self._wm_subscribers[key] = (receiver, session_id, generation)
                pending = self._wm_send_pending.get(key)
                if pending is not None and pending[4] == generation:
                    self._wm_send_pending[key] = (
                        receiver,
                        session_id,
                        pending[2],
                        pending[3],
                        generation,
                    )
            self._wm_send_cv.notify()

    def snapshot_wm_subscribers(self, bootstrap_info_groups) -> list:
        """Capture generation-guarded subscriber tokens for node cleanup."""
        keys = {
            tuple(str(bootstrap_info) for bootstrap_info in bootstrap_infos)
            for bootstrap_infos in bootstrap_info_groups
            if bootstrap_infos
        }
        with self._wm_send_cv:
            return [
                (key, self._wm_subscribers[key][2])
                for key in keys
                if key in self._wm_subscribers
            ]

    def unregister_wm_subscribers(self, subscriber_tokens, reason: str) -> None:
        """Remove only subscribers whose captured generation is still active."""
        removed = []
        with self._wm_send_cv:
            for key, generation in subscriber_tokens:
                active = self._wm_subscribers.get(key)
                if active is None or active[2] != generation:
                    continue
                self._wm_subscribers.pop(key, None)
                pending = self._wm_send_pending.get(key)
                if pending is not None and pending[4] == generation:
                    self._wm_send_pending.pop(key, None)
                token = (key, generation)
                self._wm_send_wait_started.pop(token, None)
                self._wm_send_last_log.pop(token, None)
                self._wm_send_error_last_log.pop(token, None)
                removed.append((key, active[1], generation))
            self._wm_send_cv.notify()
        for key, session_id, generation in removed:
            logger.info(
                "STAGING_WATERMARK_SUBSCRIBER_REMOVE subscriber=%s "
                "session=%s generation=%s reason=%s",
                key,
                session_id,
                generation,
                reason,
            )

    def _queue_watermark_broadcast(self, wm_round: int, wm_tail: int) -> None:
        """Queue only the newest watermark for every known prefill peer."""
        with self._wm_send_cv:
            for key, (
                receiver,
                session_id,
                generation,
            ) in self._wm_subscribers.items():
                self._wm_send_pending[key] = (
                    receiver,
                    session_id,
                    wm_round,
                    wm_tail,
                    generation,
                )
            self._wm_send_cv.notify()

    def _requeue_watermark(self, key, item) -> bool:
        """Retry a failed send without replacing a newer queued watermark."""
        with self._wm_send_cv:
            active = self._wm_subscribers.get(key)
            generation = item[4]
            if active is None or active[2] != generation:
                return False
            retry_item = (active[0], active[1], item[2], item[3], generation)
            current = self._wm_send_pending.get(key)
            if (
                current is None
                or current[4] != generation
                or (current[2], current[3]) < (item[2], item[3])
            ):
                self._wm_send_pending[key] = retry_item
            self._wm_send_cv.notify()
            return True

    def _watermark_sender_loop(self) -> None:
        """Send watermarks off-thread without blocking on ZMQ backpressure."""
        import zmq

        while True:
            with self._wm_send_cv:
                while not self._wm_send_pending:
                    self._wm_send_cv.wait()
                pending = list(self._wm_send_pending.items())
                self._wm_send_pending.clear()

            retry_needed = False
            for key, item in pending:
                receiver, session_id, wm_round, wm_tail, generation = item
                with self._wm_send_cv:
                    active = self._wm_subscribers.get(key)
                    if active is None or active[2] != generation:
                        continue
                    # Use the freshest request-scoped receiver for this peer.
                    receiver, session_id, generation = active
                token = (key, generation)
                message = [
                    b"WATERMARK",
                    str(wm_round).encode("ascii"),
                    str(wm_tail).encode("ascii"),
                    session_id.encode("ascii"),
                ]
                delivered = True
                for bootstrap_info in receiver.bootstrap_infos:
                    lock = None
                    acquired = False
                    try:
                        sock, lock = receiver._connect_to_bootstrap_server(
                            bootstrap_info
                        )
                        acquired = lock.acquire(timeout=0.01)
                        if not acquired:
                            delivered = False
                            continue
                        sock.send_multipart(message, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        delivered = False
                    except Exception:
                        delivered = False
                        now = time.monotonic()
                        last_error_log = self._wm_send_error_last_log.get(token, 0.0)
                        if now - last_error_log >= STAGING_WATERMARK_SEND_ERROR_EVERY_S:
                            logger.exception(
                                "STAGING_WATERMARK_SEND_ERROR subscriber=%s "
                                "session=%s generation=%s watermark=(%s,%s)",
                                key,
                                session_id,
                                generation,
                                wm_round,
                                wm_tail,
                            )
                            self._wm_send_error_last_log[token] = now
                    finally:
                        if acquired and lock is not None:
                            lock.release()

                if delivered:
                    self._wm_send_wait_started.pop(token, None)
                    self._wm_send_last_log.pop(token, None)
                    self._wm_send_error_last_log.pop(token, None)
                    continue

                if not self._requeue_watermark(key, item):
                    continue
                retry_needed = True
                now = time.monotonic()
                started = self._wm_send_wait_started.setdefault(token, now)
                last_log = self._wm_send_last_log.get(token, 0.0)
                if (
                    now - started >= STAGING_WATERMARK_SEND_WARN_AFTER_S
                    and now - last_log >= STAGING_WATERMARK_SEND_WARN_EVERY_S
                ):
                    logger.warning(
                        "STAGING_WATERMARK_SEND_BLOCKED subscriber=%s "
                        "session=%s generation=%s watermark=(%s,%s) waited=%.1fs; "
                        "coalescing and retrying off scheduler thread",
                        key,
                        session_id,
                        generation,
                        wm_round,
                        wm_tail,
                        now - started,
                    )
                    self._wm_send_last_log[token] = now

            if retry_needed:
                time.sleep(STAGING_WATERMARK_SEND_RETRY_S)

    def num_writers_for(self, decode_req) -> int:
        """Compute num_writers for a specific request based on its prefill TP."""
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        if prefill_tp > self.decode_tp:
            return prefill_tp // max(1, self.decode_tp)
        return 1

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
        )

    # ------------------------------------------------------------------
    # Registration: called from main thread (DecodeTransferQueue)
    # ------------------------------------------------------------------

    def _outstanding_alloc_ids(self, decode_req, receiver) -> set[int]:
        alloc_ids = {
            info[0]
            for info in getattr(receiver, "chunk_staging_infos", [])
            if info[0] >= 0
        }
        for item in getattr(decode_req, "_chunk_events", []):
            alloc_ids.add(item[1])
        last_alloc_id = getattr(decode_req, "_scatter_alloc_id", -1)
        if last_alloc_id >= 0:
            alloc_ids.add(last_alloc_id)
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

    def fence_failed_room(self, room: int, reason: str) -> None:
        """Fence late scatter before the scheduler releases request KV pages."""
        alloc_ids = self._terminalize_room(room, reason)
        if not alloc_ids:
            return
        # The quarantine callback has already armed the independent 0-60s
        # process exit. A stuck CUDA drain therefore cannot suppress fail-stop.
        stream = self.staging_allocator._scatter_stream
        if stream is not None:
            stream.synchronize()

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
        decode_req._staging_scatter_done = False
        decode_req._chunk_events = []
        decode_req._staging_last_scatter_submitted = False
        decode_req._scatter_event = None
        decode_req._scatter_alloc_id = -1
        decode_req._scatter_chunk_idx = -1
        # Stall clock: None means "no allocations held / progress just made".
        # It starts counting the first time advance_scatter observes the room
        # holding staging allocations, and is reset by every progress event
        # (chunk arrival, scatter submit, event completion). Progress is
        # recorded by the decode thread while timeout checks run on the
        # scheduler thread, so serialize the timestamp and failure decision.
        decode_req._staging_progress_lock = threading.Lock()
        decode_req._staging_stall_since = None
        decode_req._staging_stall_failed = False
        self._room_lifecycles[room] = StagingRoomLifecycle()
        self._room_to_decode_req[room] = decode_req
        self._room_to_receiver[room] = decode_req.kv_receiver

    def unregister_decode_req(self, room: int) -> None:
        # The lifecycle terminal flag closes new work before the maps disappear.
        # This lets release_room still discover and quarantine every live alloc.
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
        # Quarantine it before draining scatter; the timer is armed first so a
        # stuck CUDA synchronize cannot prevent process-level fail-stop.
        quarantined = self._terminalize_room(room, "room-release")
        stream = self.staging_allocator._scatter_stream
        if quarantined and stream is not None:
            stream.synchronize()

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
        is_last_chunk: bool = False,
    ) -> bool:
        """Submit scatter for an intermediate chunk whose writers all arrived.

        Called from decode_thread.  Records a CUDA event on decode_req so
        the main thread can later check completion and free the allocation.
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
                        if is_last_chunk:
                            decode_req._scatter_event = event
                            decode_req._scatter_alloc_id = alloc_id
                            decode_req._scatter_chunk_idx = chunk_idx
                            decode_req._staging_last_scatter_submitted = True
                        else:
                            decode_req._chunk_events.append(
                                (event, alloc_id, chunk_idx)
                            )
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
        engine_rank: int,
        peer_name: str,
        chunk_writer_counts: dict,
    ) -> Tuple[bool, bool]:
        """Validate and record a staging arrival from any transport.

        Accumulates writer arrivals in *chunk_writer_counts* and submits scatter
        once all writers for this chunk have reported in. Returns True if scatter
        was submitted.
        """
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
            if state != "WRITABLE" or expected_geometry != (
                page_start,
                num_pages,
            ):
                lifecycle.terminal = True
                violation = (
                    f"[STAGING_GEOMETRY] notification mismatch room={room} "
                    f"chunk={chunk_idx} state={state} got={(page_start, num_pages)} "
                    f"expected={expected_geometry}"
                )
            else:
                num_writers = self.num_writers_for(decode_req)
                prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
                writer_slot = (engine_rank % prefill_tp) % num_writers
                if writer_slot not in range(num_writers):
                    lifecycle.terminal = True
                    violation = (
                        f"[STAGING_WRITER] invalid slot room={room} chunk={chunk_idx} "
                        f"engine_rank={engine_rank} slot={writer_slot} "
                        f"expected=0..{num_writers - 1}"
                    )
                else:
                    seen = lifecycle.seen_writer_slots.setdefault(chunk_idx, set())
                    if writer_slot in seen:
                        logger.warning(
                            "[STAGING_DUPLICATE_WRITER] room=%s chunk=%s "
                            "engine_rank=%s writer_slot=%s peer=%s",
                            room,
                            chunk_idx,
                            engine_rank,
                            writer_slot,
                            peer_name,
                        )
                        return (False, False)
                    seen.add(writer_slot)
                    chunk_writer_counts[room][chunk_idx].add(writer_slot)
                    self._note_staging_progress(decode_req)
                    return (True, len(seen) == num_writers)
        logger.error(violation)
        self.fail_staging_room(room, violation)
        return (False, False)

    def submit_last_scatter_async(self, room: int) -> bool:
        """Submit scatter for the last chunk when all ranks report Success.

        Called from decode_thread.  Sets ``_scatter_event`` **before**
        ``_staging_last_scatter_submitted`` so the main thread sees the
        event when it checks the flag (CPython GIL guarantees ordering).
        """
        lifecycle = self._room_lifecycles.get(room)
        if lifecycle is None:
            logger.warning(
                "[STAGING] submit_last_scatter_async: room=%s not registered. "
                "This should not happen if register_decode_req is called at "
                "kv_receiver.init() time.",
                room,
            )
            return False
        with lifecycle.lock:
            if not lifecycle.chunk_geometry:
                decode_req = self._room_to_decode_req.get(room)
                if decode_req is not None:
                    decode_req._staging_scatter_done = True
                    return True
                return False
            chunk_idx = max(lifecycle.chunk_geometry)
            page_start, num_pages = lifecycle.chunk_geometry[chunk_idx]
            decode_req = self._room_to_decode_req.get(room)
            if decode_req is None or len(
                lifecycle.seen_writer_slots.get(chunk_idx, set())
            ) < self.num_writers_for(decode_req):
                return False
            state = lifecycle.chunk_states.get(chunk_idx)
            if state in ("SCATTER_SUBMITTED", "SCATTER_DONE"):
                return True
        return self.submit_chunk_scatter(
            room,
            chunk_idx,
            page_start,
            num_pages,
            is_last_chunk=True,
        )

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
                            self._free_and_send_watermark(alloc_id, decode_req)

        if getattr(decode_req, "_staging_last_scatter_submitted", False):
            event = getattr(decode_req, "_scatter_event", None)
            if event is not None and event.query():
                with lifecycle.lock:
                    self._note_staging_progress(decode_req)
                    alloc_id = decode_req._scatter_alloc_id
                    chunk_idx = decode_req._scatter_chunk_idx
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
                        self._free_and_send_watermark(alloc_id, decode_req)
                    decode_req._scatter_event = None
                    decode_req._scatter_alloc_id = -1
                    decode_req._scatter_chunk_idx = -1
                    decode_req._staging_scatter_done = True

        # Consume completed events before checking the watchdog. Otherwise a
        # last-scatter event that completes on the timeout boundary can be
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
        if getattr(decode_req, "_scatter_event", None) is not None:
            return True
        chunk_infos = (
            getattr(receiver, "chunk_staging_infos", []) if receiver is not None else []
        )
        return any(info[0] >= 0 for info in chunk_infos)

    def _note_staging_progress(self, decode_req: DecodeRequest) -> None:
        """Atomically refresh a room's stall clock after real progress."""
        with decode_req._staging_progress_lock:
            # Once the scheduler has committed the failure decision, late
            # notifications must not revive the room.
            if not decode_req._staging_stall_failed:
                decode_req._staging_stall_since = None

    def _check_room_stall(self, room: int, decode_req: DecodeRequest) -> None:
        """Fail a room that holds ring allocations but makes no progress.

        Runs on the scheduler main thread from advance_scatter. Failing the
        room through the receiver status (same mechanism as
        _check_waiting_timeout) routes it into the existing
        Failed -> pop_transferred -> unregister_decode_req -> release_room
        path, which frees its allocations and unpins the shared ring.
        The clock only runs while allocations are held, so queued rooms that
        have not engaged staging yet are never failed here.
        """
        receiver = self._room_to_receiver.get(room)
        now = time.monotonic()
        with decode_req._staging_progress_lock:
            if decode_req._staging_stall_failed:
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

    def _free_and_send_watermark(
        self, alloc_id: int, decode_req: DecodeRequest
    ) -> None:
        """Free a staging allocation and asynchronously broadcast its watermark."""
        if not self.staging_allocator.free(alloc_id):
            return
        wm_round, wm_tail = self.staging_allocator.get_watermark()
        self._queue_watermark_broadcast(wm_round, wm_tail)


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
        elif violation is None:
            result = staging_allocator.assign(required)
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            if result is None:
                logger.error(
                    "[STAGING_REQ] alloc failed room=%s chunk=%d (need %d bytes, "
                    "buffer total=%d bytes). Increase "
                    "SGLANG_DISAGG_STAGING_POOL_SIZE_MB.",
                    room,
                    chunk_idx,
                    required,
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
                alloc_id, offset, rnd = result
                end = offset + required
                infos[chunk_idx] = (alloc_id, offset, rnd, end, chunk_num_pages)
                lifecycle.chunk_states[chunk_idx] = "WRITABLE"

    if violation is not None:
        logger.error(violation)
        staging_handler.fail_staging_room(room, violation)
        return

    bootstrap_infos = room_bootstrap.get(room)
    if bootstrap_infos:
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
