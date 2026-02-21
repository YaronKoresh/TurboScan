import sys
from unittest.mock import MagicMock, patch

from turboscan.hardware.config import HardwareConfig, detect_hardware


class TestHardwareConfig:
    def test_detect_hardware_full_capabilities(self) -> None:
        """Test hardware detection on a high-end machine (GPU + psutil)."""

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        prop = MagicMock()
        prop.total_memory = 12 * 1024**3
        prop.name = "Tesla T4"
        mock_torch.cuda.get_device_properties.return_value = prop

        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value.total = 32 * 1024**3
        mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

        with patch.dict(
            sys.modules, {"torch": mock_torch, "psutil": mock_psutil}
        ):
            config = detect_hardware()

            assert config.cpu_count_physical == 4
            assert config.memory_total == 32 * 1024**3
            assert config.gpu_count == 2
            assert config.gpu_names == ["Tesla T4", "Tesla T4"]

    def test_detect_hardware_minimal(self) -> None:
        """Test fallback logic when dependencies are missing."""

        MagicMock(side_effect=ImportError("Not found"))

        with patch("os.cpu_count", return_value=4):
            with patch.dict(sys.modules, {"torch": None, "psutil": None}):
                config = detect_hardware()

                assert config.gpu_count == 0
                assert config.gpu_names == []

                assert config.memory_total == 8 * 1024**3

    def test_hardware_config_defaults(self) -> None:
        """Ensure the dataclass has safe defaults."""
        config = HardwareConfig()
        assert config.cpu_count >= 1
        assert isinstance(config.gpu_memory, list)
