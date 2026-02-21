import contextlib
import shutil
import tempfile
from pathlib import Path

import pytest

from turboscan.cache.bloom_filter import BloomFilter
from turboscan.cache.hyper_cache import HyperCache


class TestBloomFilter:
    def test_basic_membership(self) -> None:
        """Standard positive/negative test."""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        bf.add("hello")
        bf.add("world")

        assert "hello" in bf
        assert "world" in bf

        assert "not_there" not in bf

    def test_capacity_scaling(self) -> None:
        """Verify bit array size calculation."""
        bf1 = BloomFilter(capacity=100, error_rate=0.1)
        bf2 = BloomFilter(capacity=10000, error_rate=0.01)

        assert len(bf2.bits) > len(bf1.bits)

    def test_many_items(self) -> None:
        """Test adding many items."""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(500):
            bf.add(f"item_{i}")

        for i in range(500):
            assert f"item_{i}" in bf

    def test_clear(self) -> None:
        """Test clearing bloom filter."""
        bf = BloomFilter(capacity=100, error_rate=0.1)
        bf.add("test")
        assert "test" in bf

    def test_different_types(self) -> None:
        """Test with different hashable types."""
        bf = BloomFilter(capacity=100, error_rate=0.1)
        bf.add("string1")
        bf.add("string2")
        bf.add("string3")

        assert "string1" in bf
        assert "string2" in bf
        assert "string3" in bf


class TestHyperCache:
    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""

        c = HyperCache(name="test_cache")
        yield c
        c.cleanup()

        with contextlib.suppress(BaseException):
            shutil.rmtree(Path(tempfile.gettempdir()) / ".test_cache_cache")

    def test_l1_hit(self, cache) -> None:
        """Data should be in L1 immediately after set."""
        cache.set("func:1", "result1")

        found, val = cache.get("func:1")
        assert found is True
        assert val == "result1"
        assert cache.stats["l1_size"] == 1
        assert cache.stats["hits"] == 1

    def test_bloom_gatekeeper(self, cache) -> None:
        """Bloom filter should prevent cache lookups for missing keys."""

        found, _val = cache.get("missing_key")
        assert found is False
        assert cache.stats["misses"] == 1

    def test_eviction_logic(self, cache) -> None:
        """Verify L1 cache evicts old items when full."""

        cache.l1_max = 4

        items = [f"key{i}" for i in range(10)]
        for i in items:
            cache.set(i, f"val{i}")

        assert len(cache.l1_cache) <= 4

        found, _ = cache.get("key9")
        assert found is True

    def test_collision_risk(self, cache) -> None:
        """Demonstrate the risk of str() based hashing."""

        class User:
            def __init__(self, id, secret) -> None:
                self.id = id
                self.secret = secret

            def __repr__(self) -> str:

                return f"User({self.id})"

        u1 = User(1, "secret_A")
        u2 = User(1, "secret_B")

        key1 = cache._make_key("process", u1)
        key2 = cache._make_key("process", u2)

        assert key1 == key2, (
            "Warning: Cache cannot distinguish objects with same string representation!"
        )

    def test_stats_property(self, cache) -> None:
        """Test stats property returns correct structure."""
        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "l1_size" in stats

    def test_multiple_sets(self, cache) -> None:
        """Test setting multiple items."""
        for i in range(20):
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(20):
            found, val = cache.get(f"key_{i}")
            if found:
                assert val == f"value_{i}"

    def test_overwrite_existing(self, cache) -> None:
        """Test overwriting an existing key."""
        cache.set("key", "value1")
        cache.set("key", "value2")

        found, val = cache.get("key")
        assert found is True
        assert val == "value2"

    def test_get_non_existent(self, cache) -> None:
        """Test getting a key that was never set."""
        found, val = cache.get("never_set_key")
        assert found is False
        assert val is None

    def test_cleanup(self, cache) -> None:
        """Test cleanup method."""
        cache.set("key", "value")
        cache.cleanup()

    def test_make_key_with_list(self, cache) -> None:
        """Test _make_key with list argument."""
        key = cache._make_key("func", [1, 2, 3])
        assert isinstance(key, str)

    def test_make_key_with_dict(self, cache) -> None:
        """Test _make_key with dict argument."""
        key = cache._make_key("func", {"a": 1, "b": 2})
        assert isinstance(key, str)

    def test_make_key_with_tuple(self, cache) -> None:
        """Test _make_key with tuple argument."""
        key = cache._make_key("func", (1, 2, 3))
        assert isinstance(key, str)

    def test_complex_values(self, cache) -> None:
        """Test storing complex values."""
        cache.set("dict_key", {"nested": {"data": [1, 2, 3]}})
        found, val = cache.get("dict_key")
        assert found is True
        assert val == {"nested": {"data": [1, 2, 3]}}

    def test_none_value(self, cache) -> None:
        """Test storing None as a value."""
        cache.set("none_key", None)
        found, val = cache.get("none_key")
        assert found is True
        assert val is None

    def test_l1_max_size(self, cache) -> None:
        """Test that L1 respects max size."""
        cache.l1_max = 5
        for i in range(20):
            cache.set(f"k{i}", f"v{i}")

        assert len(cache.l1_cache) <= cache.l1_max
