import sys
from unittest.mock import ANY, MagicMock, patch

import pytest

from turboscan.__main__ import main


class TestTurboScanMain:
    @pytest.fixture
    def mock_deps(self):
        """Mock all external dependencies used in __main__."""
        with patch("turboscan.__main__.HyperExecutor") as mock_exec, patch(
            "turboscan.__main__.HyperAuditor"
        ) as mock_audit, patch(
            "turboscan.__main__.console"
        ) as mock_console, patch(
            "turboscan.__main__.HARDWARE"
        ) as mock_hardware, patch(
            "turboscan.__main__.HyperBoost"
        ) as mock_boost:
            mock_hardware.cpu_count = 8
            mock_hardware.memory_total = 16 * 1024**3
            mock_hardware.gpu_names = ["Tesla T4"]
            mock_hardware.gpu_memory = [8 * 1024**3]

            yield mock_exec, mock_audit, mock_console, mock_hardware, mock_boost

    def test_info_command_with_rich(self, mock_deps) -> None:
        """Test 'info' command when Rich is available."""
        _, _, mock_console, _, _ = mock_deps

        with patch.object(sys, "argv", ["turboscan", "info"]), patch(
            "turboscan.__main__.RICH_AVAIL", True
        ), patch("turboscan.__main__.GPU_AVAIL", True):
            main()

            mock_console.print.assert_called()
            args = mock_console.print.call_args[0]
            assert "Table" in str(type(args[0]))

    def test_info_command_without_rich(self, mock_deps) -> None:
        """Test 'info' command when Rich is NOT available (fallback to print)."""
        _, _, _, _, _ = mock_deps

        with patch.object(sys, "argv", ["turboscan", "info"]), patch(
            "turboscan.__main__.RICH_AVAIL", False
        ), patch("builtins.print") as mock_print:
            main()

            assert mock_print.call_count > 0

            call_args = [str(c) for c in mock_print.call_args_list]
            assert any("CPUs" in c for c in call_args)
            assert any("Memory" in c for c in call_args)

    def test_default_command_is_info(self, mock_deps) -> None:
        """Running without arguments should default to 'info' behavior."""
        _, _, mock_console, _, _ = mock_deps

        with patch.object(sys, "argv", ["turboscan"]), patch(
            "turboscan.__main__.RICH_AVAIL", True
        ):
            main()
            mock_console.print.assert_called()

    def test_audit_command_success(self, mock_deps) -> None:
        """Test 'audit' command with successful result."""
        _, mock_audit_cls, _, _, _ = mock_deps

        mock_audit_cls.return_value.run.return_value = (
            0,
            5,
        )

        with patch.object(
            sys, "argv", ["turboscan", "audit", "./src", "--exclude", "temp"]
        ), pytest.raises(SystemExit) as exc:
            main()

        mock_audit_cls.assert_called_with(ANY, {"temp"}, check_unused=True)

        assert exc.value.code == 0

    def test_audit_command_failure(self, mock_deps) -> None:
        """Test 'audit' command when errors are found."""
        _, mock_audit_cls, _, _, _ = mock_deps

        mock_audit_cls.return_value.run.return_value = (3, 0)

        with patch.object(
            sys, "argv", ["turboscan", "audit", "--no-unused"]
        ), pytest.raises(SystemExit) as exc:
            main()

        mock_audit_cls.assert_called_with(ANY, ANY, check_unused=False)

        assert exc.value.code == 1

    def test_run_command_defaults(self, mock_deps) -> None:
        """Test 'run' command with default arguments."""
        mock_exec_cls, _, _, _, _ = mock_deps

        with patch.object(
            sys, "argv", ["turboscan", "run", "script.py", "arg1"]
        ):
            main()

            mock_exec_cls.assert_called_with(
                "script.py",
                audit_first=True,
                optimize=True,
                inject_jit=True,
            )

            mock_exec_cls.return_value.execute.assert_called()

            assert sys.argv == ["script.py", "arg1"]

    def test_run_command_flags(self, mock_deps) -> None:
        """Test 'run' command with all flags enabled/disabled."""
        mock_exec_cls, _, _, _, mock_boost = mock_deps

        cmd = [
            "turboscan",
            "run",
            "--fast",
            "--no-optimize",
            "--no-jit",
            "--debug",
            "script.py",
        ]

        with patch.object(sys, "argv", cmd):
            main()

            mock_exec_cls.assert_called_with(
                "script.py", audit_first=False, optimize=False, inject_jit=False
            )

            mock_boost.set_debug.assert_called_with(True)

    def test_help_display(self) -> None:
        """Test that help is printed when unknown command is used."""
        with patch("argparse.ArgumentParser.print_help") as mock_help:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(command="weird_cmd")
                main()
                mock_help.assert_called()
