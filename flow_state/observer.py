"""
SplineObserver — monitors capture directories and extracts feature traces.

Watches a directory for new JSON files, extracts features (entropy, complexity
metrics), and writes structured trace files for downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import Feature, Trace

logger = logging.getLogger("flow_state.observer")


def _shannon_entropy(data: dict[str, Any]) -> float:
    """Calculate normalized Shannon entropy of a JSON-serializable dict.

    Uses the frequency of unique characters in the canonical JSON string
    as a proxy for information content. Returns a value in [0, 1].
    """
    text = json.dumps(data, sort_keys=True, default=str)
    if not text or len(text) < 2:
        return 0.0
    # An empty or trivially-empty container has no information
    if data is None or (isinstance(data, (dict, list)) and len(data) == 0):
        return 0.0
    # Count character frequencies
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    # Normalize by max possible entropy (log2 of unique chars)
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


class SplineObserver:
    """Monitors a capture directory and produces feature-extracted traces.

    Parameters
    ----------
    capture_dir:
        Directory to watch for incoming JSON capture files.
    trace_dir:
        Directory where output trace files are written.
    observer_id:
        Identifier for this observer instance (included in traces).
    provenance:
        Optional dict written into each trace's provenance field.
    feature_extractor:
        Optional callable ``(data: dict) -> Feature`` to override the
        default feature extraction.  When *None*, the built-in entropy-based
        extractor is used.
    poll_interval:
        Seconds between polling cycles in :meth:`run` (default 10).
    """

    def __init__(
        self,
        capture_dir: str | Path,
        trace_dir: str | Path,
        observer_id: str = "SplineObserver",
        *,
        provenance: dict[str, str] | None = None,
        feature_extractor: Callable[[dict[str, Any]], Feature] | None = None,
        poll_interval: float = 10.0,
    ) -> None:
        self.capture_dir = Path(capture_dir)
        self.trace_dir = Path(trace_dir)
        self.observer_id = observer_id
        self.provenance = provenance or {}
        self._extractor = feature_extractor or self._default_extractor
        self.poll_interval = poll_interval
        self.processed_files: set[Path] = set()

        self.trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SplineObserver '%s' monitoring %s", observer_id, self.capture_dir)

    # -- public API --------------------------------------------------

    def run_cycle(self) -> int:
        """Process one polling cycle. Returns number of new traces written."""
        capture_files = sorted(self.capture_dir.rglob("*.json"))
        new_traces = 0
        for cap_path in capture_files:
            if cap_path in self.processed_files:
                continue
            logger.info("New capture detected: %s", cap_path.name)
            trace = self.analyze_capture(cap_path)
            if trace:
                self._write_trace(trace, cap_path.stem)
                self.processed_files.add(cap_path)
                new_traces += 1
        return new_traces

    def run(self) -> None:
        """Run the observer loop indefinitely."""
        while True:
            self.run_cycle()
            time.sleep(self.poll_interval)

    def analyze_capture(self, json_path: str | Path) -> Trace | None:
        """Read a capture JSON and produce a :class:`Trace`."""
        json_path = Path(json_path)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Error reading %s: %s", json_path, exc)
            return None

        feature = self._extractor(data)
        return Trace(
            observer_id=self.observer_id,
            timestamp=data.get("ts", datetime.now().isoformat()),
            source_capture=json_path.name,
            features=feature,
            provenance={
                "origin_system": self.provenance.get("origin_system", "generic"),
                "capture_mode": self.provenance.get("capture_mode", "automated"),
                **{k: v for k, v in self.provenance.items() if k not in ("origin_system", "capture_mode")},
            },
        )

    # -- internals ---------------------------------------------------

    def _default_extractor(self, data: dict[str, Any]) -> Feature:
        """Built-in feature extractor using Shannon entropy."""
        entropy = _shannon_entropy(data)
        # Derive a few lightweight metrics from the data
        text = json.dumps(data, sort_keys=True, default=str)
        field_count = len(data) if isinstance(data, dict) else 0
        return Feature(
            visual_density=min(len(text) / 4096.0, 1.0),
            signal_noise_ratio=1.0 - entropy,  # inverse: more predictable = higher SNR
            momentum_vector=float(field_count) / 20.0,
            entropy=round(entropy, 6),
        )

    def _write_trace(self, trace: Trace, stem: str) -> Path:
        trace_name = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{stem}.json"
        trace_path = self.trace_dir / trace_name
        with open(trace_path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        logger.info("Trace written: %s", trace_name)
        return trace_path
