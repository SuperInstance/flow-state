"""Tests for flow_state.models — data model serialization, defaults, and logic."""

import pytest
from datetime import datetime
from pathlib import Path
from flow_state.models import Feature, Trace, Anomaly, Manifest


# ─── Feature tests ──────────────────────────────────────────────

class TestFeatureDefaults:
    def test_all_defaults_zero(self):
        f = Feature()
        assert f.visual_density == 0.0
        assert f.signal_noise_ratio == 0.0
        assert f.momentum_vector == 0.0
        assert f.entropy == 0.0

    def test_construction_with_values(self):
        f = Feature(visual_density=0.5, signal_noise_ratio=0.8, momentum_vector=-0.3, entropy=1.2)
        assert f.visual_density == 0.5
        assert f.signal_noise_ratio == 0.8
        assert f.momentum_vector == -0.3
        assert f.entropy == 1.2


class TestFeatureSerialization:
    def test_to_dict_roundtrip(self):
        f = Feature(0.1, 0.2, 0.3, 0.4)
        d = f.to_dict()
        assert d == {"visual_density": 0.1, "signal_noise_ratio": 0.2, "momentum_vector": 0.3, "entropy": 0.4}
        f2 = Feature.from_dict(d)
        assert f == f2

    def test_from_dict_missing_keys_defaults_to_zero(self):
        f = Feature.from_dict({})
        assert f.visual_density == 0.0
        assert f.signal_noise_ratio == 0.0
        assert f.momentum_vector == 0.0
        assert f.entropy == 0.0

    def test_from_dict_partial_data(self):
        f = Feature.from_dict({"visual_density": 0.7})
        assert f.visual_density == 0.7
        assert f.signal_noise_ratio == 0.0
        assert f.entropy == 0.0

    def test_from_dict_ignores_extra_keys(self):
        f = Feature.from_dict({"visual_density": 0.5, "extra_key": "ignored"})
        assert f.visual_density == 0.5


# ─── Trace tests ────────────────────────────────────────────────

class TestTraceDefaults:
    def test_construction(self):
        f = Feature(entropy=0.5)
        t = Trace(observer_id="obs1", timestamp="2026-01-01T00:00:00", source_capture="cap.json", features=f)
        assert t.observer_id == "obs1"
        assert t.timestamp == "2026-01-01T00:00:00"
        assert t.source_capture == "cap.json"
        assert t.features == f
        assert t.provenance == {}
        assert t.trace_path is None

    def test_provenance_default_empty_dict(self):
        t = Trace("obs", "ts", "src", Feature())
        assert t.provenance == {}

    def test_trace_path_default_none(self):
        t = Trace("obs", "ts", "src", Feature())
        assert t.trace_path is None


class TestTraceSerialization:
    def test_to_dict_roundtrip(self):
        f = Feature(0.1, 0.2, 0.3, 0.4)
        t = Trace(
            observer_id="observer_42",
            timestamp="2026-08-08T20:00:00",
            source_capture="capture_001.json",
            features=f,
            provenance={"model": "granite", "version": "3.1"},
        )
        d = t.to_dict()
        t2 = Trace.from_dict(d)
        assert t2.observer_id == "observer_42"
        assert t2.timestamp == "2026-08-08T20:00:00"
        assert t2.source_capture == "capture_001.json"
        assert t2.features == f
        assert t2.provenance == {"model": "granite", "version": "3.1"}

    def test_from_dict_missing_observer_defaults_to_unknown(self):
        t = Trace.from_dict({})
        assert t.observer_id == "unknown"

    def test_from_dict_missing_source_defaults_to_empty(self):
        t = Trace.from_dict({})
        assert t.source_capture == ""

    def test_from_dict_missing_features_defaults_to_zero(self):
        t = Trace.from_dict({})
        assert t.features == Feature()

    def test_from_dict_missing_provenance_defaults_to_empty(self):
        t = Trace.from_dict({})
        assert t.provenance == {}

    def test_from_dict_with_trace_path(self):
        path = Path("/tmp/trace.json")
        t = Trace.from_dict({"observer_id": "x"}, trace_path=path)
        assert t.trace_path == path

    def test_to_dict_does_not_include_trace_path(self):
        t = Trace("obs", "ts", "src", Feature(), trace_path=Path("/tmp/x"))
        d = t.to_dict()
        assert "trace_path" not in d


# ─── Anomaly tests ──────────────────────────────────────────────

class TestAnomalyConstruction:
    def test_construction(self):
        a = Anomaly(measured_entropy=0.9, baseline_mean=0.3, deviation_score=0.6, source_trace="trace.json")
        assert a.measured_entropy == 0.9
        assert a.baseline_mean == 0.3
        assert a.deviation_score == 0.6
        assert a.source_trace == "trace.json"

    def test_timestamp_auto_generated(self):
        a = Anomaly(measured_entropy=0.9, baseline_mean=0.3, deviation_score=0.6, source_trace="t.json")
        # Should be a valid ISO timestamp
        parsed = datetime.fromisoformat(a.timestamp)
        assert isinstance(parsed, datetime)


class TestAnomalySeverity:
    @pytest.mark.parametrize("score,expected", [
        (0.51, "critical"),
        (0.6, "critical"),
        (0.9, "critical"),
        (1.0, "critical"),
        (0.26, "high"),
        (0.4, "high"),
        (0.49, "high"),
        (0.5, "high"),  # exactly 0.5 → not > 0.5 → high
        (0.25, "moderate"),  # exactly 0.25 → not > 0.25 → moderate
        (0.1, "moderate"),
        (0.0, "moderate"),
        (-0.5, "moderate"),  # negative scores should be moderate
    ])
    def test_severity_thresholds(self, score, expected):
        a = Anomaly(measured_entropy=0.5, baseline_mean=0.3, deviation_score=score, source_trace="t.json")
        assert a.severity == expected

    def test_critical_boundary_strict_greater(self):
        a = Anomaly(measured_entropy=0.5, baseline_mean=0.3, deviation_score=0.5, source_trace="t.json")
        # 0.5 is NOT > 0.5, so it's "high" not "critical"
        assert a.severity == "high"

    def test_high_boundary_strict_greater(self):
        a = Anomaly(measured_entropy=0.5, baseline_mean=0.3, deviation_score=0.25, source_trace="t.json")
        # 0.25 is NOT > 0.25, so it's "moderate" not "high"
        assert a.severity == "moderate"


class TestAnomalySerialization:
    def test_to_dict_roundtrip_fields(self):
        a = Anomaly(measured_entropy=0.8, baseline_mean=0.2, deviation_score=0.6, source_trace="trace.json")
        d = a.to_dict()
        assert d["measured_entropy"] == 0.8
        assert d["baseline_mean"] == 0.2
        assert d["deviation_score"] == 0.6
        assert d["source_trace"] == "trace.json"
        assert "timestamp" in d


# ─── Manifest tests ─────────────────────────────────────────────

class TestManifest:
    def test_construction(self):
        anomaly = Anomaly(measured_entropy=0.9, baseline_mean=0.3, deviation_score=0.6, source_trace="t.json")
        m = Manifest(
            manifest_id="man_001",
            timestamp="2026-08-08T20:00:00",
            source_trace="t.json",
            anomaly=anomaly,
            training_payload={"data": "payload"},
        )
        assert m.manifest_id == "man_001"
        assert m.timestamp == "2026-08-08T20:00:00"
        assert m.source_trace == "t.json"
        assert m.anomaly == anomaly
        assert m.training_payload == {"data": "payload"}
        assert m.manifest_path is None

    def test_manifest_path_default_none(self):
        anomaly = Anomaly(0.5, 0.3, 0.2, "t.json")
        m = Manifest("id", "ts", "src", anomaly, {})
        assert m.manifest_path is None

    def test_to_dict(self):
        anomaly = Anomaly(measured_entropy=0.9, baseline_mean=0.3, deviation_score=0.6, source_trace="t.json")
        m = Manifest(
            manifest_id="man_001",
            timestamp="2026-08-08T20:00:00",
            source_trace="t.json",
            anomaly=anomaly,
            training_payload={"key": "value"},
        )
        d = m.to_dict()
        assert d["manifest_id"] == "man_001"
        assert d["timestamp"] == "2026-08-08T20:00:00"
        assert d["source_trace"] == "t.json"
        assert d["anomaly_metrics"]["deviation_score"] == 0.6
        assert d["training_payload"] == {"key": "value"}

    def test_to_dict_with_complex_payload(self):
        anomaly = Anomaly(0.5, 0.3, 0.2, "t.json")
        payload = {
            "samples": [1, 2, 3],
            "metadata": {"model": "test", "epochs": 10},
            "config": None,
        }
        m = Manifest("id", "ts", "src", anomaly, payload)
        d = m.to_dict()
        assert d["training_payload"] == payload

    def test_to_dict_does_not_include_manifest_path(self):
        anomaly = Anomaly(0.5, 0.3, 0.2, "t.json")
        m = Manifest("id", "ts", "src", anomaly, {}, manifest_path=Path("/tmp/m.json"))
        d = m.to_dict()
        assert "manifest_path" not in d
