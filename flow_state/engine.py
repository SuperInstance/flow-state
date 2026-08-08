"""
LearningEngine — consumes trace files and flags entropy anomalies.

Reads JSON traces produced by :class:`~flow_state.observer.SplineObserver`,
maintains a rolling baseline of entropy values, and emits manifest files when
an anomaly is detected (entropy exceeds mean + N×std).
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import Anomaly, Manifest, Trace

logger = logging.getLogger("flow_state.engine")


class LearningEngine:
    """Process traces and detect entropy anomalies.

    Parameters
    ----------
    trace_dir:
        Directory containing JSON trace files to process.
    manifest_dir:
        Directory where anomaly manifests are written.
    entropy_threshold:
        Number of standard deviations above the mean to flag as anomaly
        (default 2.0 — i.e. mean + 2σ).
    rolling_window:
        Maximum number of recent entropy values kept for baseline
        calculation (default 50).
    min_baseline:
        Minimum number of observations before anomaly detection kicks in
        (default 10).
    poll_interval:
        Seconds between polling cycles in :meth:`run` (default 10).
    """

    def __init__(
        self,
        trace_dir: str | Path,
        manifest_dir: str | Path,
        *,
        entropy_threshold: float = 2.0,
        rolling_window: int = 50,
        min_baseline: int = 10,
        poll_interval: float = 10.0,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.manifest_dir = Path(manifest_dir)
        self.entropy_threshold = entropy_threshold
        self.rolling_window = rolling_window
        self.min_baseline = min_baseline
        self.poll_interval = poll_interval
        self.processed_traces: set[Path] = set()
        self.entropy_history: list[float] = []

        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "LearningEngine monitoring %s (threshold=%.1fσ, window=%d)",
            self.trace_dir,
            entropy_threshold,
            rolling_window,
        )

    # -- public API --------------------------------------------------

    def run_cycle(self) -> int:
        """Process all unprocessed traces. Returns anomaly count."""
        trace_files = sorted(self.trace_dir.glob("*.json"))
        new_flags = 0
        for trace_path in trace_files:
            if trace_path in self.processed_traces:
                continue
            try:
                trace = self._load_trace(trace_path)
            except Exception as exc:
                logger.error("Error processing trace %s: %s", trace_path, exc)
                continue

            if trace is None:
                continue

            entropy = trace.features.entropy
            self.entropy_history.append(entropy)
            if len(self.entropy_history) > self.rolling_window:
                self.entropy_history.pop(0)

            mean_e, std_e = self.calculate_baseline()
            is_anomaly = False
            if len(self.entropy_history) >= self.min_baseline:
                if entropy > (mean_e + self.entropy_threshold * std_e):
                    is_anomaly = True

            if is_anomaly:
                anomaly = Anomaly(
                    measured_entropy=entropy,
                    baseline_mean=mean_e,
                    deviation_score=entropy - mean_e,
                    source_trace=trace_path.name,
                )
                self._create_manifest(trace, anomaly)
                new_flags += 1

            self.processed_traces.add(trace_path)
        return new_flags

    def run(self) -> None:
        """Run the engine loop indefinitely."""
        while True:
            self.run_cycle()
            time.sleep(self.poll_interval)

    def calculate_baseline(self) -> tuple[float, float]:
        """Return (mean, std) of the rolling entropy history."""
        if len(self.entropy_history) < 5:
            return 0.0, 0.0
        arr = np.array(self.entropy_history)
        return float(np.mean(arr)), float(np.std(arr))

    # -- internals ---------------------------------------------------

    def _load_trace(self, trace_path: Path) -> Trace | None:
        with open(trace_path, "r") as f:
            data = json.load(f)
        return Trace.from_dict(data, trace_path=trace_path)

    def _create_manifest(self, trace: Trace, anomaly: Anomaly) -> Manifest:
        manifest_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        manifest = Manifest(
            manifest_id=manifest_id,
            timestamp=datetime.now().isoformat(),
            source_trace=anomaly.source_trace,
            anomaly=anomaly,
            training_payload=trace.to_dict(),
        )
        manifest_path = self.manifest_dir / f"anomaly_{manifest_id}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.warning(
            "ANOMALY DETECTED — entropy=%.4f, baseline=%.4f, σ-deviation=%.2f — manifest: %s",
            anomaly.measured_entropy,
            anomaly.baseline_mean,
            anomaly.deviation_score,
            manifest_path.name,
        )
        manifest.manifest_path = manifest_path
        return manifest
