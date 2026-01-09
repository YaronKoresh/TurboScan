import pytest
from unittest.mock import MagicMock, patch
from turboscan.gpu.accelerator import GPUAccelerator

class TestGPUAccelerator:
    
    @patch('turboscan.gpu.accelerator.GPU_AVAIL', False)
    def test_init_no_gpu(self):
        """Should handle systems without GPU gracefully."""
        acc = GPUAccelerator()
        assert acc.device is None
        assert acc.streams == []

    @patch('turboscan.gpu.accelerator.GPU_AVAIL', True)
    @patch('turboscan.gpu.accelerator.TORCH_AVAIL', True)
    @patch('turboscan.gpu.accelerator.GPU_COUNT', 4)
    @patch('turboscan.gpu.accelerator.torch')
    def test_init_with_gpu(self, mock_torch):
        """Should select CUDA device and create streams if available."""
        # We need to ensure torch.cuda.Stream works
        mock_torch.cuda.Stream.return_value = MagicMock()
        
        acc = GPUAccelerator()
        
        # Should create streams (min(8, 4*4) = 8 streams)
        assert len(acc.streams) > 0
        assert mock_torch.cuda.Stream.called

    @patch('turboscan.gpu.accelerator.GPU_AVAIL', True)
    @patch('turboscan.gpu.accelerator.TORCH_AVAIL', True)
    @patch('turboscan.gpu.accelerator.GPU_COUNT', 4)
    @patch('turboscan.gpu.accelerator.torch')
    def test_stream_context_manager(self, mock_torch):
        """Test the context manager for stream synchronization."""
        # Mock the stream object
        mock_stream_obj = MagicMock()
        mock_torch.cuda.Stream.return_value = mock_stream_obj
        
        acc = GPUAccelerator()
        
        # Enter context
        with acc.stream_context(idx=0):
            pass
        
        # Verify it switched to that stream
        mock_torch.cuda.stream.assert_called_with(mock_stream_obj)
        # Verify it synchronized on exit
        mock_stream_obj.synchronize.assert_called()

    def test_batch_hash_cpu(self):
        """Verify batch hashing logic works without GPU."""
        acc = GPUAccelerator()
        items = ["a", "b"]
        hashes = acc.batch_hash(items)
        assert len(hashes) == 2
        assert isinstance(hashes[0], str)