"""Quantum SHHG - HBT / g^(2) acquisition and analysis package.

Importing this package is safe on machines without the lab software: all
hardware dependencies (TimeTagger, pythonnet, elliptec, ...) are imported
lazily inside the methods that use them.
"""

from .acquisition import (
    AcquisitionConfig,
    LaserParams,
    TimeTaggerParams,
    ChunkingParams,
    ScanPoint,
    Recorder,
    RawTimeTagRecorder,
    Acquisition,
)
from .hardware import (
    TimeTaggerDevice,
    RotationStage,
    PRM1Stage,
    ELL14Stage,
    RotationStageController,
)
from .measurement import (
    CorrelationRecorder,
    coincidence_threshold_stop,
)

__all__ = [
    "AcquisitionConfig",
    "LaserParams",
    "TimeTaggerParams",
    "ChunkingParams",
    "ScanPoint",
    "Recorder",
    "RawTimeTagRecorder",
    "Acquisition",
    "TimeTaggerDevice",
    "RotationStage",
    "PRM1Stage",
    "ELL14Stage",
    "RotationStageController",
    "CorrelationRecorder",
    "coincidence_threshold_stop",
]
