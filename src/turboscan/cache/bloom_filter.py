"""Probabilistic membership testing with Bloom filters."""

import hashlib
from typing import List


class BloomFilter:
    def __init__(
        self, capacity: int = 100000, error_rate: float = 0.01
    ) -> None:
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_count = self._optimal_bits(capacity, error_rate)
        self.hash_count = self._optimal_hashes(self.bit_count, capacity)
        self.bits = bytearray((self.bit_count + 7) // 8)

    @staticmethod
    def _optimal_bits(n: int, p: float) -> int:
        import math

        return int(-n * math.log(p) / math.log(2) ** 2)

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        import math

        return max(1, int(m / n * math.log(2)))

    def _hashes(self, item: str) -> List[int]:
        # Use Blake2b for faster hashing (2-3x faster than MD5/SHA1)
        item_bytes = item.encode()
        h1 = int.from_bytes(
            hashlib.blake2b(item_bytes, digest_size=16).digest(), "big"
        )
        h2 = int.from_bytes(
            hashlib.blake2b(item_bytes, digest_size=16, key=b"salt").digest(),
            "big",
        )
        return [(h1 + i * h2) % self.bit_count for i in range(self.hash_count)]

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self.bits[pos // 8] |= 1 << pos % 8

    def __contains__(self, item: str) -> bool:
        # Optimized to short-circuit on first miss
        return all(
            self.bits[pos // 8] & 1 << pos % 8 for pos in self._hashes(item)
        )
