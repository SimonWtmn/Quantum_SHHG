"""
=============================================================================
Acquisition Orchestration
=============================================================================

The acquisition layer of the HBT / g^(2) pipeline. 
    connect hardware -> configure -> (optionally move a stage) -> record for a
    duration -> save -> repeat / scan / chunk-until-enough -> tear down.

What is written to disk for each acquisition is decided by a pluggable :class:`Recorder`:

* :class:`RawTimeTagRecorder` (this module) streams the raw photon time tags to
  .ttbin and is the genuinely "strict acquisition" default - no correlation
  or coincidence computation happens here.
* A live-reduction recorder (correlations / coincidences / heralding) lives in a
  separate processing helper (src/measurement.py).

This separation keeps acquisition lightweight and lets the heavy analysis code
evolve independently while sharing one driver, one parameter schema and one folder/logging convention.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
Date: 19/06/2026
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .hardware import TimeTaggerDevice, RotationStageController, StageID

__all__ = [
    "LaserParams",
    "TimeTaggerParams",
    "ChunkingParams",
    "AcquisitionConfig",
    "ScanPoint",
    "Recorder",
    "RawTimeTagRecorder",
    "Acquisition",
    "StopCondition",
]

StopCondition = Callable[[dict], "tuple[bool, str]"]


# =============================================================================
# Small IO / utility helpers
# =============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands the common numpy scalar / array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, filepath: Union[str, Path]):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=_NumpyEncoder)


def save_pickle(data, filepath: Union[str, Path]):
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_time_date() -> "tuple[str, str]":
    """Return (HH-MM-SS, YYYY-MM-DD) for filenames and metadata."""
    now = datetime.now()
    return now.strftime("%H-%M-%S"), now.strftime("%Y-%m-%d")


def setup_logger(save_dir: Path, name: str = "acquisition") -> logging.Logger:
    """File + console logger writing ``<save_dir>/<save_dir.name>.log``."""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(save_dir / f"{save_dir.name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger


# =============================================================================
# Configuration (dataclasses)
# =============================================================================

@dataclass
class LaserParams:
    """Driving-laser metadata (not controlled here, only recorded)."""
    rep_rate_hz: float = 18.8e6
    wavelength_nm: float = 2100.0


@dataclass
class TimeTaggerParams:
    """Everything needed to configure the TimeTagger and describe the channels.

    The field names mirror the schema consumed by :class:`src.hbt_core.HBTMeasurement`
    so a recorder's output is directly analysable downstream.
    """
    tt_mode: str = "Standard"
    channels: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    mode_on_channel: Sequence[str] = field(
        default_factory=lambda: ["H3T", "H3R", "H4T", "H4R", "H5T", "H5R"])
    trigger_levels_v: Union[float, Sequence[float]] = 0.5
    deadtime_ps: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    delays_ps: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    binwidth_ps: int = 100
    num_bins: int = 5000
    coincidence_window_ps: int = 1000
    trigger: Sequence[int] = field(default_factory=list)
    filter: Sequence[int] = field(default_factory=list)

    def to_schema(self) -> dict:
        """Dict in the canonical ``timetagging`` shape used across the project."""
        return {
            "tt_mode": self.tt_mode,
            "binwidth_ps": self.binwidth_ps,
            "num_bins": self.num_bins,
            "channels": list(self.channels),
            "mode_on_channel": list(self.mode_on_channel),
            "deadtime": list(self.deadtime_ps),
            "delays": list(self.delays_ps),
            "coincidence_window": self.coincidence_window_ps,
            "trigger": list(self.trigger),
            "filter": list(self.filter),
        }


@dataclass
class ChunkingParams:
    """Controls chunked "measure until good enough" acquisition.

    When :attr:`enabled`, an acquisition is split into chunks of
    :attr:`chunk_minutes`. After every :attr:`check_every_n_chunks` chunks the
    accumulated result is passed to the orchestrator's stop condition; if it is
    satisfied (and :attr:`stop_when_reached`), acquisition ends early.
    :attr:`max_chunks` bounds the total duration.
    """
    enabled: bool = False
    chunk_minutes: float = 2.0
    max_chunks: int = 100
    check_every_n_chunks: int = 1
    stop_when_reached: bool = True


@dataclass
class AcquisitionConfig:
    """Top-level acquisition configuration."""
    base_dir: Union[str, Path]
    material: str = "Unknown"
    experiment_type: str = "raw"
    laser: LaserParams = field(default_factory=LaserParams)
    timetagging: TimeTaggerParams = field(default_factory=TimeTaggerParams)
    chunking: ChunkingParams = field(default_factory=ChunkingParams)
    repeats: int = 1
    tt_serial: str = ""
    custom: dict = field(default_factory=dict)

    def general_params(self, date: str, time_hms: str, save_dir: Path) -> dict:
        """Build the ``general`` parameter block (project-wide schema)."""
        return {
            "date": date,
            "time": time_hms,
            "experiment_type": self.experiment_type,
            "save_dir": str(save_dir),
            "material": self.material,
            "laser": asdict(self.laser),
            "timetagging": self.timetagging.to_schema(),
            "custom": dict(self.custom),
        }


@dataclass
class ScanPoint:
    """One setting of the experiment within a scan.

    ``power_mw`` and ``angle_deg`` are recorded as metadata; if both
    ``angle_deg`` and ``stage_id`` are given the orchestrator moves that stage
    to the angle before acquiring.
    """
    power_mw: Optional[float] = None
    angle_deg: Optional[float] = None
    stage_id: Optional[StageID] = None
    label: Optional[str] = None
    extra: dict = field(default_factory=dict)


# =============================================================================
# Recorders (the acquisition <-> processing seam)
# =============================================================================

class Recorder(ABC):
    """Strategy that defines *what* a single acquisition writes to disk.

    Implementations receive the configured :class:`TimeTaggerDevice` and a
    duration, build whatever TimeTagger measurements they need, run them, save
    their result and return a JSON/pickle-friendly ``dict``. Keeping this
    interface tiny is what lets a future live-correlation processing recorder
    drop in beside :class:`RawTimeTagRecorder` without touching the orchestrator.
    """

    name: str = "recorder"

    @abstractmethod
    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        """Acquire for ``duration_ps`` and persist; return a result dict."""

    def merge(self, accumulated: Optional[dict], new: dict) -> dict:
        """Accumulate chunk results for chunked acquisition.

        The default sums integer counters, sums durations and duration-weights
        any ``countrates`` mapping. Recorders with richer payloads (e.g.
        correlation histograms) should override this.
        """
        if accumulated is None:
            merged = dict(new)
            merged.setdefault("files", [])
            if "file" in new and new["file"] not in merged["files"]:
                merged["files"] = [new["file"]]
            return merged

        merged = dict(accumulated)
        d_acc = accumulated.get("duration_ps", 0) or 0
        d_new = new.get("duration_ps", 0) or 0
        total = d_acc + d_new or 1

        for key in ("total_events", "total_size"):
            if key in new:
                merged[key] = accumulated.get(key, 0) + new[key]
        merged["duration_ps"] = d_acc + d_new

        for key in ("counts",):
            if key in new:
                acc_map = accumulated.get(key, {})
                merged[key] = {k: acc_map.get(k, 0) + v for k, v in new[key].items()}

        if "countrates" in new:
            acc_r = accumulated.get("countrates", {})
            merged["countrates"] = {
                k: (acc_r.get(k, 0.0) * d_acc + v * d_new) / total
                for k, v in new["countrates"].items()
            }

        files = list(accumulated.get("files", []))
        if "file" in new:
            files.append(new["file"])
        merged["files"] = files
        return merged


class RawTimeTagRecorder(Recorder):
    """Stream raw time tags to a ``.ttbin`` file (the strict-acquisition default).

    Optionally runs a parallel :class:`Counter`/``Countrate`` on the channels so
    that singles counts/rates are available for logging and for count-based
    early-stop conditions, all without any correlation processing.

    Parameters
    ----------
    max_file_size_mb : float
        Maximum size per ``.ttbin`` subfile (0 = single file / driver default).
    record_countrate : bool
        If True (default) attach a Countrate to report singles.
    """

    name = "raw"

    def __init__(self, max_file_size_mb: float = 100.0, record_countrate: bool = True):
        self.max_file_size_mb = max_file_size_mb
        self.record_countrate = record_countrate

    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        import TimeTagger as TT

        tagger = device._require()
        channels = list(params["general"]["timetagging"]["channels"])
        ttbin_path = str(savepath.with_suffix(".ttbin"))

        sync = TT.SynchronizedMeasurements(tagger)
        fw = TT.FileWriter(sync.getTagger(), ttbin_path, channels)
        if self.max_file_size_mb > 0:
            fw.setMaxFileSize(int(self.max_file_size_mb * 1024 * 1024))

        countrate = None
        if self.record_countrate:
            countrate = TT.Countrate(sync.getTagger(), channels)

        logger.info("Recording raw tags -> %s (%.1f s)", ttbin_path, duration_ps * 1e-12)
        sync.startFor(int(duration_ps))
        sync.waitUntilFinished()

        result = {
            "recorder": self.name,
            "file": ttbin_path,
            "duration_ps": int(duration_ps),
            "total_events": int(fw.getTotalEvents()),
            "total_size": int(fw.getTotalSize()),
        }
        if countrate is not None:
            counts = countrate.getCountsTotal()
            rates = countrate.getData()
            result["counts"] = {str(ch): int(counts[i]) for i, ch in enumerate(channels)}
            result["countrates"] = {str(ch): float(rates[i]) for i, ch in enumerate(channels)}

        del fw, countrate, sync

        data = {"Parameters": params, "data": result}
        save_pickle(data, str(savepath) + ".meta.pkl")
        logger.info("Raw acquisition done: %d events, %.2f MB.",
                    result["total_events"], result["total_size"] / (1024 * 1024))
        return result


# =============================================================================
# Orchestrator
# =============================================================================

class Acquisition:
    """Drive single, scanned and chunked acquisitions over the configured hardware.

    Parameters
    ----------
    config : AcquisitionConfig
        Static experiment configuration.
    recorder : Recorder, optional
        What to write each acquisition (defaults to :class:`RawTimeTagRecorder`).
        Pass a live-processing recorder here to enable on-the-fly reduction.
    device : TimeTaggerDevice, optional
        Pre-built tagger wrapper; one is created from ``config`` if omitted.
    stages : RotationStageController, optional
        Connected stages for power/angle scans (optional).
    stop_condition : StopCondition, optional
        Callable evaluated on the accumulated result during chunked runs.

    Notes
    -----
    Typical lifecycle::

        acq = Acquisition(config, recorder=RawTimeTagRecorder())
        acq.setup()                       # connect + configure
        acq.run_scan(points)              # or run_point / run_chunked
        acq.teardown()

    or simply ``with Acquisition(config) as acq: acq.run_scan(points)``.
    """

    def __init__(self, config: AcquisitionConfig,
                 recorder: Optional[Recorder] = None,
                 device: Optional[TimeTaggerDevice] = None,
                 stages: Optional[RotationStageController] = None,
                 stop_condition: Optional[StopCondition] = None):
        self.config = config
        self.recorder = recorder or RawTimeTagRecorder()
        self.stages = stages
        self.stop_condition = stop_condition

        self.time_hms, self.date = get_time_date()
        self.save_dir = Path(config.base_dir) / f"{self.date}_{self.time_hms}_{config.experiment_type}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.save_dir, name="acquisition")

        self.device = device or TimeTaggerDevice(
            serial=config.tt_serial, resolution=config.timetagging.tt_mode,
            logger=self.logger)
        self._general_params = self.config.general_params(self.date, self.time_hms, self.save_dir)

    # -- setup / teardown -----------------------------------------------------

    def setup(self) -> "Acquisition":
        """Connect the tagger, validate channels and apply the configuration."""
        self.logger.info("=" * 20 + " ACQUISITION SETUP " + "=" * 20)
        save_json(self._general_params, self.save_dir / "general_parameters.json")

        if not self.device.is_connected:
            self.device.connect()

        tt = self.config.timetagging
        if not self.device.validate_channels(tt.channels):
            raise RuntimeError("Requested channels are not all available; aborting.")

        self.device.set_trigger_levels(tt.channels, tt.trigger_levels_v)
        if any(tt.deadtime_ps):
            self.device.set_dead_times(tt.channels, tt.deadtime_ps)
        if any(tt.delays_ps):
            self.device.set_input_delays(tt.channels, tt.delays_ps)
        if tt.trigger and tt.filter:
            self.device.set_conditional_filter(tt.trigger, tt.filter)
        self.logger.info("Recorder: %s", self.recorder.name)
        return self

    def teardown(self):
        """Free the tagger and disconnect any stages."""
        self.device.free()
        if self.stages is not None:
            self.stages.disconnect_all()
        self.logger.info("Acquisition teardown complete.")

    # -- optional characterisation -------------------------------------------

    def characterize(self, label: str, duration_s: float = 5.0,
                     prompt: Optional[str] = None) -> dict:
        """Measure per-channel singles count rate (e.g. background vs signal).

        If ``prompt`` is given, wait for the user (cover/uncover the signal)
        before measuring. Returns the {channel: counts/s} mapping and logs it.
        """
        if prompt:
            input(prompt)
        self.logger.info("Characterising '%s' for %.0f s ...", label, duration_s)
        rates = self.device.measure_countrate(self.config.timetagging.channels, duration_s)
        for ch, r in rates.items():
            self.logger.info("  %s ch %d: %.0f counts/s", label, ch, r)
        return rates

    # -- single point ---------------------------------------------------------

    def _experiment_params(self, duration_ps: int, savepath: Path,
                           point: ScanPoint) -> dict:
        return {
            "type": self.config.experiment_type,
            "duration": int(duration_ps),
            "savepath": str(savepath),
            "laser_power": point.power_mw,
            "rotation_stage": point.angle_deg,
            **point.extra,
        }

    def _move_stage_for(self, point: ScanPoint):
        if point.angle_deg is None or self.stages is None:
            return
        self.stages.move_to(point.angle_deg, point.stage_id)
        self.logger.info("Stage %s -> %.4f deg", point.stage_id, point.angle_deg)

    def run_point(self, point: ScanPoint, duration_s: float,
                  file_stem: Optional[str] = None) -> dict:
        """Acquire a single fixed-duration measurement at ``point``."""
        self._move_stage_for(point)
        duration_ps = int(duration_s * 1e12)
        stem = file_stem or self._default_stem(point)
        savepath = self.save_dir / stem
        params = {"general": self._general_params,
                  "experimental": self._experiment_params(duration_ps, savepath, point)}
        self.logger.info("Single acquisition '%s' (%.2f min)", stem, duration_s / 60)
        return self.recorder.record(self.device, duration_ps, savepath,
                                    params=params, logger=self.logger)

    # -- chunked --------------------------------------------------------------

    def run_chunked(self, point: ScanPoint, file_stem: Optional[str] = None) -> dict:
        """Acquire in chunks, merging into a running total until the stop
        condition is met or ``max_chunks`` is reached."""
        self._move_stage_for(point)
        ch = self.config.chunking
        chunk_ps = int(ch.chunk_minutes * 60e12)
        stem = file_stem or self._default_stem(point)

        merged_dir = self.save_dir / "MERGED"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_fp = merged_dir / f"{stem}_MERGED.pkl"

        self.logger.info("Chunked acquisition '%s' (chunk=%.2f min, max=%d).",
                         stem, ch.chunk_minutes, ch.max_chunks)

        def _merged_payload(n_done: int) -> dict:
            mp_params = copy.deepcopy(self._general_params)
            exp = self._experiment_params(chunk_ps * n_done, merged_fp, point)
            return {"Parameters": {"general": mp_params, "experimental": exp},
                    "data": merged}

        merged: Optional[dict] = None
        for i in range(ch.max_chunks):
            chunk_stem = f"{stem}_chunk{i}"
            savepath = self.save_dir / chunk_stem
            params = {"general": self._general_params,
                      "experimental": self._experiment_params(chunk_ps, savepath, point)}
            self.logger.info("Chunk %d/%d | elapsed %.1f min",
                             i + 1, ch.max_chunks, (i + 1) * ch.chunk_minutes)

            result = self.recorder.record(self.device, chunk_ps, savepath,
                                          params=params, logger=self.logger)
            merged = self.recorder.merge(merged, result)

            if (i + 1) % ch.check_every_n_chunks == 0:
                save_pickle(_merged_payload(i + 1), merged_fp)
                self.logger.info("Updated merged total -> %s", merged_fp)

                if self.stop_condition is not None:
                    stop, reason = self.stop_condition(merged)
                    self.logger.info("Stop check: %s", reason)
                    if stop and ch.stop_when_reached:
                        self.logger.info(">>> STOP CONDITION REACHED - ending early <<<")
                        save_pickle(_merged_payload(i + 1), merged_fp)
                        self.logger.info("Final merged data -> %s", merged_fp)
                        return merged  # leave early; we have what we need

        save_pickle(_merged_payload(ch.max_chunks), merged_fp)
        self.logger.info("Final merged data -> %s", merged_fp)
        return merged

    # -- scan -----------------------------------------------------------------

    def run_scan(self, points: Sequence[ScanPoint]) -> None:
        """Run every ``point`` (single or chunked per config) ``repeats`` times."""
        for num_repeat in range(self.config.repeats):
            for point in points:
                t0 = datetime.now()
                stem = self._default_stem(point, num_repeat)
                if self.config.chunking.enabled:
                    self.run_chunked(point, file_stem=stem)
                else:
                    self.run_point(point, duration_s=self._default_single_duration_s(), file_stem=stem)
                dt_min = (datetime.now() - t0).total_seconds() / 60
                self.logger.info("Point '%s' completed in %.2f min", stem, dt_min)

    # -- helpers --------------------------------------------------------------

    def _default_single_duration_s(self) -> float:
        return self.config.chunking.chunk_minutes * 60.0

    def _default_stem(self, point: ScanPoint, num_repeat: int = 0) -> str:
        if point.label:
            base = point.label
        else:
            p = f"{point.power_mw:g}mW" if point.power_mw is not None else "run"
            base = f"{self.time_hms}_{self.date}_{p}"
        return f"{base}_num{num_repeat}"

    def __enter__(self) -> "Acquisition":
        return self.setup()

    def __exit__(self, exc_type, exc, tb):
        self.teardown()
        return False
