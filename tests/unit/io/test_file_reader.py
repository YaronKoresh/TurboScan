import pytest
import os
import asyncio
from pathlib import Path
from turboscan.io.file_reader import FastFileReader

# Check if pytest-asyncio is available
try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False

class TestFastFileReader:
    @pytest.fixture
    def reader(self):
        # Use 2 workers to test parallelism without spawning too many threads
        return FastFileReader(max_workers=2)

    def test_read_small_file_standard(self, reader, tmp_path):
        """Small files (<64KB) should be read via standard IO."""
        f = tmp_path / "small.txt"
        content = "Hello TurboScan"
        f.write_text(content, encoding="utf-8")
        
        assert reader.read_file(f) == content

    def test_read_large_file_mmap(self, reader, tmp_path):
        """Large files (>64KB) should be read via mmap."""
        # Create 65KB file
        content = "a" * (65 * 1024) 
        f = tmp_path / "large.txt"
        f.write_text(content, encoding="utf-8")
        
        assert reader.read_file(f) == content

    def test_read_missing_file_returns_empty(self, reader):
        """Missing files should return empty string, not crash."""
        assert reader.read_file(Path("non_existent_ghost.py")) == ""

    def test_read_files_parallel(self, reader, tmp_path):
        """Verify threaded batch reading."""
        files = []
        expected = {}
        for i in range(3):
            p = tmp_path / f"file_{i}.txt"
            content = f"content_{i}"
            p.write_text(content, encoding="utf-8")
            files.append(p)
            expected[p] = content
            
        results = reader.read_files_parallel(files)
        
        assert len(results) == 3
        for p, content in results.items():
            assert expected[p] == content

    @pytest.mark.skipif(not PYTEST_ASYNCIO_AVAILABLE, reason="pytest-asyncio not available")
    @pytest.mark.asyncio
    async def test_read_files_async(self, reader, tmp_path):
        """Verify async batch reading."""
        p = tmp_path / "async.txt"
        p.write_text("async_data", encoding="utf-8")
        
        results = await reader.read_files_async([p])
        assert results[p] == "async_data"