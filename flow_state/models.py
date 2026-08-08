"""Data models for flow-state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Feature:
    """A single feature extracted from a capture file."""

    visual_density: float = 0.0
    signal_noise_ratio: float = 0.0
    momentum_vector: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "visual_density": self.visual_density,
            "signal_noise_ratio": self.signal_noise_ratio,
            "momentum_vector": self.momentum_vector,
            "entropy": self.entropy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feature:
        return cls(
            visual_density=data.get("visual_density", 0.0),
            signal_noise_ratio=data.get("signal_noise_ratio", 0.0),
            momentum_vector=data.get("momentum_vector", 0.0),
            entropy=data.get("entropy", 0.0),
        )


@dataclass
class Trace:
    """A structured observation trace written by SplineObserver."""

    observer_id: str
    timestamp: str
    source_capture: str
    features: Feature
    provenance: dict[str, str] = field(default_factory=dict)
    trace_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "timestamp": self.timestamp,
            "source_capture": self.source_capture,
            "features": self.features.to_dict(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], trace_path: Path | None = None) -> Trace:
        return cls(
            observer_id=data.get("observer_id", "unknown"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            source_capture=data.get("source_capture", ""),
            features=Feature.from_dict(data.get("features", {})),
            provenance=data.get("provenance", {}),
            trace_path=trace_path,
        )


@dataclass
class Anomaly:
    """An anomaly detected by the LearningEngine."""

    measured_entropy: float
    baseline_mean: float
    deviation_score: float
    source_trace: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def severity(self) -> str:
        """Qualitative severity label."""
        if self.deviation_score > 0.5:
            return "critical"
        if self.deviation_score > 0.25:
            return "high"
        return "moderate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_entropy": self.measured_entropy,
            "baseline_mean": self.baseline_mean,
            "deviation_score": self.deviation_score,
            "source_trace": self.source_trace,
            "timestamp": self.timestamp,
        }


@dataclass
class Manifest:
    """A training manifest produced when an anomaly is flagged."""

    manifest_id: str
    timestamp: str
    source_trace: str
    anomaly: Anomaly
    training_payload: dict[str, Any]
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "timestamp": self.timestamp,
            "source_trace": self.source_trace,
            "anomaly_metrics": self.anomaly.to_dict(),
            "training_payload": self.training_payload,
        }
