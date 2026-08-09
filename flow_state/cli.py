"""Flow-state CLI entry point.

Usage::

    flow-state observe <capture-dir> --trace-dir <dir>
    flow-state analyze <trace-dir> --manifest-dir <dir>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .engine import LearningEngine
from .observer import SplineObserver


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """Entropy-based stream observation and anomaly detection."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s — %(message)s",
    )


@main.command()
@click.argument("capture_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--trace-dir", "-t", type=click.Path(path_type=Path), required=True, help="Output directory for traces.")
@click.option("--observer-id", "-i", default="SplineObserver", help="Observer identifier.")
@click.option("--interval", "-n", type=float, default=None, help="Run once (no flag) or set poll interval in seconds for continuous mode.")
@click.option("--once", is_flag=True, help="Run a single cycle and exit.")
def observe(
    capture_dir: Path,
    trace_dir: Path,
    observer_id: str,
    interval: float | None,
    once: bool,
) -> None:
    """Watch a capture directory and produce feature traces."""
    observer = SplineObserver(
        capture_dir=capture_dir,
        trace_dir=trace_dir,
        observer_id=observer_id,
        poll_interval=interval or 10.0,
    )
    if once or interval is None:
        # Default: run once (if --once flag, or if neither --once nor --interval given)
        count = observer.run_cycle()
        click.echo(f"Processed {count} new capture(s).")
    else:
        # Continuous mode: --interval was explicitly set without --once
        click.echo(f"Observing {capture_dir} (poll every {observer.poll_interval}s). Ctrl-C to stop.")
        observer.run()


@main.command()
@click.argument("trace_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--manifest-dir", "-m", type=click.Path(path_type=Path), required=True, help="Output directory for anomaly manifests.")
@click.option("--threshold", "-s", type=float, default=2.0, help="Std-dev multiplier for anomaly threshold (default 2.0).")
@click.option("--window", "-w", type=int, default=50, help="Rolling window size (default 50).")
@click.option("--once", is_flag=True, help="Run a single cycle and exit.")
@click.option("--interval", "-n", type=float, default=10.0, help="Poll interval in seconds for continuous mode.")
def analyze(
    trace_dir: Path,
    manifest_dir: Path,
    threshold: float,
    window: int,
    once: bool,
    interval: float,
) -> None:
    """Analyze traces and flag entropy anomalies."""
    engine = LearningEngine(
        trace_dir=trace_dir,
        manifest_dir=manifest_dir,
        entropy_threshold=threshold,
        rolling_window=window,
        poll_interval=interval,
    )
    if once:
        flags = engine.run_cycle()
        click.echo(f"Detected {flags} anomaly/anomalies.")
    else:
        click.echo(f"Analyzing {trace_dir} (poll every {interval}s). Ctrl-C to stop.")
        engine.run()


if __name__ == "__main__":
    main()
