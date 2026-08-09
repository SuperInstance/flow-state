"""Tests for the flow-state CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from flow_state.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    """Create a capture directory with a sample JSON capture."""
    d = tmp_path / "captures"
    d.mkdir()
    capture = {
        "visual_density": 0.5,
        "signal_noise_ratio": 2.0,
        "momentum_vector": 0.3,
        "entropy": 1.2,
    }
    (d / "cap001.json").write_text(json.dumps(capture))
    return d


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


@pytest.fixture
def manifest_dir(tmp_path: Path) -> Path:
    d = tmp_path / "manifests"
    d.mkdir()
    return d


# ── main group ────────────────────────────────────────────────


class TestCliMain:
    def test_help(self, runner: CliRunner):
        """CLI --help returns usage info."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "entropy" in result.output.lower() or "observation" in result.output.lower()

    def test_verbose_flag(self, runner: CliRunner):
        """--verbose flag is accepted."""
        result = runner.invoke(main, ["-v", "--help"])
        assert result.exit_code == 0


# ── observe command ───────────────────────────────────────────


class TestCliObserve:
    def test_observe_help(self, runner: CliRunner):
        """observe --help shows usage."""
        result = runner.invoke(main, ["observe", "--help"])
        assert result.exit_code == 0
        assert "capture" in result.output.lower() or "observe" in result.output.lower()

    def test_observe_once(self, runner: CliRunner, capture_dir: Path, trace_dir: Path):
        """observe --once processes captures and creates traces."""
        result = runner.invoke(main, [
            "observe", str(capture_dir),
            "--trace-dir", str(trace_dir),
            "--once",
        ])
        assert result.exit_code == 0
        assert "Processed" in result.output
        # Should have created at least one trace file
        traces = list(trace_dir.glob("*.json"))
        assert len(traces) >= 1

    def test_observe_default_runs_once(self, runner: CliRunner, capture_dir: Path, trace_dir: Path):
        """Without --once or --interval, observe runs once and exits."""
        result = runner.invoke(main, [
            "observe", str(capture_dir),
            "--trace-dir", str(trace_dir),
        ])
        assert result.exit_code == 0
        assert "Processed" in result.output

    def test_observe_with_observer_id(self, runner: CliRunner, capture_dir: Path, trace_dir: Path):
        """Custom observer ID appears in traces."""
        result = runner.invoke(main, [
            "observe", str(capture_dir),
            "--trace-dir", str(trace_dir),
            "--observer-id", "TestObserver",
            "--once",
        ])
        assert result.exit_code == 0
        traces = list(trace_dir.glob("*.json"))
        if traces:
            data = json.loads(traces[0].read_text())
            assert data.get("observer_id") == "TestObserver"

    def test_observe_nonexistent_capture_dir(self, runner: CliRunner, trace_dir: Path):
        """Nonexistent capture directory fails with error."""
        result = runner.invoke(main, [
            "observe", "/nonexistent/path/xyz",
            "--trace-dir", str(trace_dir),
            "--once",
        ])
        assert result.exit_code != 0

    def test_observe_empty_capture_dir(self, runner: CliRunner, tmp_path: Path, trace_dir: Path):
        """Empty capture directory processes 0 captures."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(main, [
            "observe", str(empty),
            "--trace-dir", str(trace_dir),
            "--once",
        ])
        assert result.exit_code == 0
        assert "0" in result.output


# ── analyze command ───────────────────────────────────────────


class TestCliAnalyze:
    def test_analyze_help(self, runner: CliRunner):
        """analyze --help shows usage."""
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "anomal" in result.output.lower() or "analyz" in result.output.lower()

    def test_analyze_once_empty_traces(self, runner: CliRunner, trace_dir: Path, manifest_dir: Path):
        """analyze --once on empty trace directory produces 0 anomalies."""
        result = runner.invoke(main, [
            "analyze", str(trace_dir),
            "--manifest-dir", str(manifest_dir),
            "--once",
        ])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_analyze_with_traces(self, runner: CliRunner, trace_dir: Path, manifest_dir: Path):
        """analyze with actual traces processes them."""
        # Create a few trace files
        for i in range(5):
            trace = {
                "observer_id": "test",
                "timestamp": f"2026-01-0{i+1}T00:00:00",
                "source_capture": f"cap{i}.json",
                "features": {
                    "visual_density": 0.5,
                    "signal_noise_ratio": 2.0,
                    "momentum_vector": 0.3,
                    "entropy": 1.0 + i * 0.01,
                },
                "provenance": {},
            }
            (trace_dir / f"trace{i}.json").write_text(json.dumps(trace))

        result = runner.invoke(main, [
            "analyze", str(trace_dir),
            "--manifest-dir", str(manifest_dir),
            "--once",
        ])
        assert result.exit_code == 0

    def test_analyze_custom_threshold(self, runner: CliRunner, trace_dir: Path, manifest_dir: Path):
        """Custom threshold is accepted."""
        result = runner.invoke(main, [
            "analyze", str(trace_dir),
            "--manifest-dir", str(manifest_dir),
            "--threshold", "3.5",
            "--once",
        ])
        assert result.exit_code == 0

    def test_analyze_custom_window(self, runner: CliRunner, trace_dir: Path, manifest_dir: Path):
        """Custom rolling window is accepted."""
        result = runner.invoke(main, [
            "analyze", str(trace_dir),
            "--manifest-dir", str(manifest_dir),
            "--window", "10",
            "--once",
        ])
        assert result.exit_code == 0

    def test_analyze_nonexistent_trace_dir(self, runner: CliRunner, manifest_dir: Path):
        """Nonexistent trace directory fails."""
        result = runner.invoke(main, [
            "analyze", "/nonexistent/path/xyz",
            "--manifest-dir", str(manifest_dir),
            "--once",
        ])
        assert result.exit_code != 0
