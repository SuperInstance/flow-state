"""
flow-state — entropy-based stream observation and anomaly detection.

Spline observers with learning engines for anomaly detection.

Public API
----------

.. code-block:: python

    from flow_state import SplineObserver, LearningEngine

    observer = SplineObserver("captures/", "traces/")
    observer.run_cycle()

    engine = LearningEngine("traces/", "manifests/")
    engine.run_cycle()
"""

from .models import Anomaly, Feature, Manifest, Trace
from .observer import SplineObserver
from .engine import LearningEngine

__version__ = "0.1.0"
__all__ = [
    "SplineObserver",
    "LearningEngine",
    "Trace",
    "Feature",
    "Anomaly",
    "Manifest",
]
