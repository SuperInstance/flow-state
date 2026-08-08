"""Tests for SplineObserver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow_state.models import Feature, Trace
from flow_state.observer import SplineObserver, _shannon_entropy


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    d = tmp_path / "captures"
    d.mkdir()
    return d


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "traces"
    d.mkdir()
    return d


def write_capture(directory: Path, name: str, data: dict) -> Path:
    p = directory / name
    p.write_text(json.dumps(data))
    return p


class TestShannonEntropy:
    def test_empty_input(self) -> None:
        assert _shannon_entropy({}) == 0.0

    def test_uniform_distribution_high_entropy(self) -> None:
        """A dict with many equally-distributed keys should have high entropy."""
        data = {f"key_{i}": f"val_{i}" for i in range(50)}
        entropy = _shannon_entropy(data)
        assert 0.0 < entropy <= 1.0
        assert entropy > 0.8  # diverse data → high entropy

    def test_repetitive_low_entropy(self) -> None:
        """A single repeated character should have near-zero entropy."""
        data = {"x": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
        entropy_repetitive = _shannon_entropy(data)
        data_diverse = {f"k{i}": chr(65 + (i % 58)) for i in range(50)}
        entropy_diverse = _shannon_entropy(data_diverse)
        assert entropy_repetitive < entropy_diverse


class TestSplineObserver:
    def test_observes_new_capture(self, capture_dir: Path, trace_dir: Path) -> None:
        write_capture(capture_dir, "cap1.json", {"ts": "2026-01-01", "value": 42})
        obs = SplineObserver(capture_dir, trace_dir, "TEST")
        count = obs.run_cycle()
        assert count == 1
        traces = list(trace_dir.glob("*.json"))
        assert len(traces) == 1

    def test_skips_already_processed(self, capture_dir: Path, trace_dir: Path) -> None:
        write_capture(capture_dir, "cap1.json", {"ts": "2026-01-01", "value": 42})
        obs = SplineObserver(capture_dir, trace_dir, "TEST")
        obs.run_cycle()
        # Second cycle should find nothing new
        count = obs.run_cycle()
        assert count == 0

    def test_analyze_capture_returns_trace(self, capture_dir: Path, trace_dir: Path) -> None:
        cap_path = write_capture(capture_dir, "cap1.json", {"ts": "2026-01-01", "data": [1, 2, 3]})
        obs = SplineObserver(capture_dir, trace_dir, "TEST")
        trace = obs.analyze_capture(cap_path)
        assert trace is not None
        assert trace.observer_id == "TEST"
        assert trace.source_capture == "cap1.json"
        assert trace.timestamp == "2026-01-01"
        assert isinstance(trace.features, Feature)
        assert 0.0 <= trace.features.entropy <= 1.0

    def test_analyze_capture_bad_json(self, capture_dir: Path, trace_dir: Path) -> None:
        cap_path = capture_dir / "bad.json"
        cap_path.write_text("not valid json{")
        obs = SplineObserver(capture_dir, trace_dir, "TEST")
        trace = obs.analyze_capture(cap_path)
        assert trace is None

    def test_custom_feature_extractor(self, capture_dir: Path, trace_dir: Path) -> None:
        write_capture(capture_dir, "cap1.json", {"value": 99})

        def extractor(data: dict) -> Feature:
            return Feature(entropy=0.5, visual_density=0.3, signal_noise_ratio=0.7, momentum_vector=1.0)

        obs = SplineObserver(capture_dir, trace_dir, "CUSTOM", feature_extractor=extractor)
        obs.run_cycle()
        traces = list(trace_dir.glob("*.json"))
        assert len(traces) == 1
        data = json.loads(traces[0].read_text())
        assert data["features"]["entropy"] == 0.5

    def test_creates_trace_dir_if_missing(self, capture_dir: Path, tmp_path: Path) -> None:
        new_trace_dir = tmp_path / "new_traces"
        assert not new_trace_dir.exists()
        obs = SplineObserver(capture_dir, new_trace_dir, "TEST")
        assert new_trace_dir.exists()

    def test_nested_json_files(self, capture_dir: Path, trace_dir: Path) -> None:
        sub = capture_dir / "subdir"
        sub.mkdir()
        write_capture(sub, "nested.json", {"ts": "2026-01-01", "val": 1})
        obs = SplineObserver(capture_dir, trace_dir, "TEST")
        count = obs.run_cycle()
        assert count == 1

    def test_trace_content_structure(self, capture_dir: Path, trace_dir: Path) -> None:
        write_capture(capture_dir, "cap1.json", {"ts": "2026-01-01", "value": 42})
        obs = SplineObserver(capture_dir, trace_dir, "TEST", provenance={"origin_system": "myapp"})
        obs.run_cycle()
        trace_file = list(trace_dir.glob("*.json"))[0]
        data = json.loads(trace_file.read_text())
        assert "observer_id" in data
        assert "timestamp" in data
        assert "source_capture" in data
        assert "features" in data
        assert "provenance" in data
        assert data["provenance"]["origin_system"] == "myapp"
