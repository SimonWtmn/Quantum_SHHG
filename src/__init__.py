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
    CombinedRecorder,
    CountrateRecorder,
    coincidence_threshold_stop,
)
from .core import HBTMeasurement
from .visu import GridVisualizer
from .powerscan import PowerScanAnalyzer, PowerScanComparison, malus_power, malus_angle
from .report import (
    AnalysisReport,
    PowerScanReport,
    PowerScanComparisonReport,
    load_run,
    discover_runs,
    discover_power_scan,
    find_run_pkl,
)

__all__ = [
    # acquisition
    "AcquisitionConfig",
    "LaserParams",
    "TimeTaggerParams",
    "ChunkingParams",
    "ScanPoint",
    "Recorder",
    "RawTimeTagRecorder",
    "Acquisition",
    # hardware
    "TimeTaggerDevice",
    "RotationStage",
    "PRM1Stage",
    "ELL14Stage",
    "RotationStageController",
    # recorders / processing
    "CorrelationRecorder",
    "CombinedRecorder",
    "CountrateRecorder",
    "coincidence_threshold_stop",
    # analysis
    "HBTMeasurement",
    "GridVisualizer",
    "PowerScanAnalyzer",
    "PowerScanComparison",
    "malus_power",
    "malus_angle",
    "AnalysisReport",
    "PowerScanReport",
    "PowerScanComparisonReport",
    "load_run",
    "discover_runs",
    "discover_power_scan",
    "find_run_pkl",
]
