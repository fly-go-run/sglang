"""GPU-free tests for the best-fit free-list staging allocator."""

import random
import threading

from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    StagingInvariantCounters,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _allocator(size=1024):
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


class TestFreeListAllocator(CustomTestCase):
    def test_grant_is_exclusive_and_round_zero(self):
        a = _allocator(1024)
        first = a.assign(256)
        second = a.assign(256)
        self.assertEqual(first, (0, 0, 0))
        self.assertEqual(second, (1, 256, 0))
        self.assertEqual(a.free_extents, [(512, 512)])
        self.assertEqual(a.get_watermark(), (0, 0))
        self.assertEqual(a.free_bytes(), 512)
        self.assertEqual(a.largest_extent(), 512)

    def test_full_returns_none_and_release_makes_room(self):
        a = _allocator(1024)
        ids = [a.assign(256)[0] for _ in range(4)]
        self.assertIsNone(a.assign(1))
        self.assertTrue(a.free(ids[1]))
        self.assertEqual(a.assign(256), (4, 256, 0))
        self.assertIsNone(a.assign(2048))  # larger than the pool

    def test_release_coalesces_both_sides(self):
        a = _allocator(1024)
        x, y, z = (a.assign(256)[0] for _ in range(3))
        a.free(x)
        a.free(z)
        self.assertEqual(a.free_extents, [(0, 256), (512, 512)])
        a.free(y)
        self.assertEqual(a.free_extents, [(0, 1024)])

    def test_best_fit_prefers_smallest_sufficient_extent(self):
        a = _allocator(1024)
        ids = [a.assign(128)[0] for _ in range(8)]
        a.free(ids[1])  # hole of 128 at 128
        a.free(ids[4])
        a.free(ids[5])  # hole of 256 at 512
        self.assertEqual(a.assign(100)[1], 128)
        self.assertEqual(a.assign(200)[1], 512)

    def test_slow_room_pins_only_its_own_extent(self):
        a = _allocator(1024)
        slow = a.assign(256)[0]
        others = [a.assign(256)[0] for _ in range(3)]
        for alloc_id in others:
            self.assertTrue(a.free(alloc_id))
        # The slow allocation is still live, yet everything else is reusable.
        self.assertEqual(a.free_bytes(), 768)
        self.assertEqual(a.largest_extent(), 768)
        self.assertEqual(a.assign(768)[1], 256)
        self.assertTrue(a.free(slow))
        self.assertEqual(a.free_extents[0], (0, 256))

    def test_quarantined_extent_never_returns(self):
        a = _allocator(1024)
        bad = a.assign(256)[0]
        self.assertTrue(a.quarantine(bad, "staging-stall"))
        self.assertFalse(a.free(bad))
        self.assertEqual(a.quarantined_bytes(), 256)
        self.assertEqual(a.free_bytes(), 768)
        self.assertEqual(a.invariant_counters.get("free_quarantined"), 1)
        self.assertIn(bad, a.allocations)

    def test_randomized_invariants(self):
        rng = random.Random(7)
        a = _allocator(4096)
        live = {}
        for _ in range(2000):
            if live and rng.random() < 0.5:
                alloc_id = rng.choice(list(live))
                self.assertTrue(a.free(alloc_id))
                live.pop(alloc_id)
            else:
                size = rng.choice([64, 128, 192, 256, 512])
                result = a.assign(size)
                if result is not None:
                    alloc_id, offset, _ = result
                    for o, s in live.values():
                        self.assertTrue(offset + size <= o or o + s <= offset)
                    live[alloc_id] = (offset, size)
            used = sum(s for _, s in live.values())
            self.assertEqual(a.free_bytes(), 4096 - used)
            extents = a.free_extents
            for (o1, s1), (o2, _) in zip(extents, extents[1:]):
                self.assertLess(o1 + s1, o2)  # sorted and never adjacent


if __name__ == "__main__":
    import unittest

    unittest.main()
