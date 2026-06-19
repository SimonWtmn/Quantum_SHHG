"""
=============================================================================
Acquisition Orchestration
=============================================================================

The acquisition layer of the HBT / g^(2) pipeline. It is responsible for the
*experiment choreography* only:

    connect hardware -> configure -> (optionally move a stage) -> record for a
    duration -> save -> repeat / scan / chunk-until-enough -> tear down.

It is deliberately **agnostic about the physics**. *What* is written to disk
for each acquisition is decided by a pluggable :class:`Recorder`:

* :class:`RawTimeTagRecorder` (this module) streams the raw photon time tags to
  ``.ttbin`` and is the genuinely "strict acquisition" default - no correlation
  or coincidence computation happens here.
* A live-reduction recorder (correlations / coincidences / heralding) lives in a
  **separate processing helper** (``src/measurement.py``) and implements the
  same :class:`Recorder` interface, so it can be handed to :class:`Acquisition`
  *only when the user asks for on-the-fly processing*. The orchestrator never
  imports it.

This separation keeps acquisition lightweight and lets the heavy analysis code
evolve independently while sharing one driver, one parameter schema and one
folder/logging convention with the rest of ``src/`` (``hbt_core``,
``hbt_powerscan``, ``hbt_visu``).

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

# Lets type hints reference a class from inside its own body (e.g. -> "Acquisition").
from __future__ import annotations

import copy       # copy.deepcopy(): make an independent copy of nested dicts
import json       # read/write JSON files (human-readable parameter dumps)
import logging    # log messages to file + console
import pickle     # save/load Python objects to/from binary .pkl files
from abc import ABC, abstractmethod                  # build the abstract Recorder base class
from dataclasses import dataclass, field, asdict     # concise "data holder" classes
from datetime import datetime                        # timestamps for filenames + timing
from pathlib import Path                             # object-oriented filesystem paths
from typing import Callable, Optional, Sequence, Union  # type hints

import numpy as np  # numeric types handled by the JSON encoder below

# Import the hardware wrappers from the sibling module (the leading dot = "this package").
from .hardware import TimeTaggerDevice, RotationStageController, StageID

# Public names of this module (what `from src.acquisition import *` exposes).
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

# A "stop condition" is any function taking the accumulated-result dict and
# returning (should_stop, reason_text). This alias documents that contract.
StopCondition = Callable[[dict], "tuple[bool, str]"]


# =============================================================================
# Small IO / utility helpers
# =============================================================================

# A custom JSON encoder. Plain json.dump cannot handle numpy numbers/arrays, so
# we teach it how by subclassing json.JSONEncoder.
class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands the common numpy scalar / array types."""

    def default(self, obj):
        # `default` is called only for objects json doesn't already know.
        if isinstance(obj, np.integer):     # numpy int -> Python int
            return int(obj)
        if isinstance(obj, np.floating):    # numpy float -> Python float
            return float(obj)
        if isinstance(obj, np.ndarray):     # numpy array -> Python list
            return obj.tolist()
        return super().default(obj)         # anything else: let the base class try (may error)


def save_json(data: dict, filepath: Union[str, Path]):
    # Open the file for writing text ("w") using UTF-8 so accents are preserved.
    with open(filepath, "w", encoding="utf-8") as f:
        # indent=4 -> pretty, readable; ensure_ascii=False -> keep real UTF-8 chars.
        json.dump(data, f, ensure_ascii=False, indent=4, cls=_NumpyEncoder)


def save_pickle(data, filepath: Union[str, Path]):
    # Open the file in binary write mode ("wb") because pickle writes bytes.
    with open(filepath, "wb") as f:
        # HIGHEST_PROTOCOL = the fastest/most compact pickle format available.
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_time_date() -> "tuple[str, str]":
    """Return (HH-MM-SS, YYYY-MM-DD) for filenames and metadata."""
    now = datetime.now()  # current local date+time
    # strftime formats the datetime into strings; "-" instead of ":" so it's filename-safe.
    return now.strftime("%H-%M-%S"), now.strftime("%Y-%m-%d")


def setup_logger(save_dir: Path, name: str = "acquisition") -> logging.Logger:
    """File + console logger writing ``<save_dir>/<save_dir.name>.log``."""
    # The format string for every log line: timestamp, level, then the message.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    logger = logging.getLogger(name)   # fetch (or create) a named logger
    if logger.hasHandlers():           # if this logger was set up before...
        logger.handlers.clear()        # ...remove old handlers to avoid duplicate lines
    logger.setLevel(logging.INFO)      # record INFO and more severe (WARNING/ERROR) messages

    # Handler 1: write log lines into a .log file inside the run folder.
    fh = logging.FileHandler(save_dir / f"{save_dir.name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Handler 2: also print log lines to the console/terminal.
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False  # don't also pass messages to the root logger (no double prints)
    return logger


# =============================================================================
# Configuration (dataclasses)
# =============================================================================

# `@dataclass` auto-writes __init__ etc. from the fields listed below, so these
# classes are concise, typed containers for settings.
@dataclass
class LaserParams:
    """Driving-laser metadata (not controlled here, only recorded)."""
    rep_rate_hz: float = 21e6       # laser repetition rate in hertz (pulses per second)
    wavelength_nm: float = 2100.0   # laser wavelength in nanometres


@dataclass
class TimeTaggerParams:
    """Everything needed to configure the TimeTagger and describe the channels.

    The field names mirror the schema consumed by :class:`src.hbt_core.HBTMeasurement`
    so a recorder's output is directly analysable downstream.
    """
    tt_mode: str = "Standard"  # resolution mode of the tagger
    # `field(default_factory=...)` is required for mutable defaults (lists/dicts),
    # so every instance gets its OWN fresh list rather than a shared one.
    channels: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])  # detector channels
    mode_on_channel: Sequence[str] = field(   # human label per channel (harmonic + arm)
        default_factory=lambda: ["H3T", "H3R", "H4T", "H4R", "H5T", "H5R"])
    trigger_levels_v: Union[float, Sequence[float]] = 0.5  # threshold voltage(s)
    deadtime_ps: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])  # dead time/ch
    delays_ps: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])    # delay/ch
    binwidth_ps: int = 100             # histogram bin width (picoseconds)
    num_bins: int = 5000               # number of bins in each correlation histogram
    coincidence_window_ps: int = 1000  # max time gap (ps) to call two tags "coincident"
    trigger: Sequence[int] = field(default_factory=list)  # channels that trigger conditional filter
    filter: Sequence[int] = field(default_factory=list)   # channels filtered by it

    def to_schema(self) -> dict:
        """Dict in the canonical ``timetagging`` shape used across the project."""
        # Re-key our attributes into the exact names the analysis code expects.
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
    enabled: bool = False          # split the acquisition into chunks at all?
    chunk_minutes: float = 2.0     # length of each chunk in minutes
    max_chunks: int = 100          # hard cap on the number of chunks (safety limit)
    check_every_n_chunks: int = 1  # evaluate the stop condition this often
    stop_when_reached: bool = True # actually stop early when the condition is met


@dataclass
class AcquisitionConfig:
    """Top-level acquisition configuration."""
    base_dir: Union[str, Path]            # parent folder where run folders are created
    material: str = "Unknown"             # sample name (recorded as metadata)
    experiment_type: str = "raw"          # tag describing the run; also the default recorder mode
    laser: LaserParams = field(default_factory=LaserParams)            # nested laser settings
    timetagging: TimeTaggerParams = field(default_factory=TimeTaggerParams)  # nested tagger settings
    chunking: ChunkingParams = field(default_factory=ChunkingParams)   # nested chunking settings
    repeats: int = 1                      # how many times to repeat the whole scan
    tt_serial: str = ""                   # specific tagger serial ("" = first available)
    custom: dict = field(default_factory=dict)  # free-form extra metadata to save

    def general_params(self, date: str, time_hms: str, save_dir: Path) -> dict:
        """Build the ``general`` parameter block (project-wide schema)."""
        # This dict is saved with every acquisition so the analysis knows the conditions.
        return {
            "date": date,
            "time": time_hms,
            "experiment_type": self.experiment_type,
            "save_dir": str(save_dir),
            "material": self.material,
            "laser": asdict(self.laser),               # dataclass -> plain dict
            "timetagging": self.timetagging.to_schema(),
            "custom": dict(self.custom),               # shallow copy so callers can't mutate ours
        }


@dataclass
class ScanPoint:
    """One setting of the experiment within a scan.

    ``power_mw`` and ``angle_deg`` are recorded as metadata; if both
    ``angle_deg`` and ``stage_id`` are given the orchestrator moves that stage
    to the angle before acquiring.
    """
    power_mw: Optional[float] = None    # laser power for this point (metadata / filename)
    angle_deg: Optional[float] = None   # rotation-stage angle to move to (or None)
    stage_id: Optional[StageID] = None  # which stage to move (serial str or address int)
    label: Optional[str] = None         # optional custom filename label
    extra: dict = field(default_factory=dict)  # any extra per-point metadata to store


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

    #: Short identifier used in logs and as the ``experiment_type`` default.
    name: str = "recorder"

    @abstractmethod
    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        """Acquire for ``duration_ps`` and persist; return a result dict."""
        # Abstract: each concrete recorder must implement its own version.

    def merge(self, accumulated: Optional[dict], new: dict) -> dict:
        """Accumulate chunk results for chunked acquisition.

        The default sums integer counters, sums durations and duration-weights
        any ``countrates`` mapping. Recorders with richer payloads (e.g.
        correlation histograms) should override this.
        """
        # First chunk: there is nothing to merge into yet, so seed the running total.
        if accumulated is None:
            merged = dict(new)                       # shallow copy of the first result
            merged.setdefault("files", [])           # ensure a "files" list exists
            if "file" in new and new["file"] not in merged["files"]:
                merged["files"] = [new["file"]]      # start the file list with this chunk's file
            return merged

        merged = dict(accumulated)                   # copy the running total to extend
        d_acc = accumulated.get("duration_ps", 0) or 0  # duration so far
        d_new = new.get("duration_ps", 0) or 0           # duration of the new chunk
        total = d_acc + d_new or 1                       # combined duration (avoid 0)

        # Integer totals simply add up.
        for key in ("total_events", "total_size"):
            if key in new:
                merged[key] = accumulated.get(key, 0) + new[key]
        merged["duration_ps"] = d_acc + d_new            # track total duration

        # Per-channel counts add up element-by-element.
        for key in ("counts",):
            if key in new:
                acc_map = accumulated.get(key, {})
                merged[key] = {k: acc_map.get(k, 0) + v for k, v in new[key].items()}

        # Count RATES are averages, so combine them weighted by each chunk's duration.
        if "countrates" in new:
            acc_r = accumulated.get("countrates", {})
            merged["countrates"] = {
                k: (acc_r.get(k, 0.0) * d_acc + v * d_new) / total
                for k, v in new["countrates"].items()
            }

        # Keep a growing list of all the raw files produced.
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

    name = "raw"  # overrides Recorder.name; appears in logs and identifies the output

    def __init__(self, max_file_size_mb: float = 100.0, record_countrate: bool = True):
        self.max_file_size_mb = max_file_size_mb  # split .ttbin into subfiles above this size
        self.record_countrate = record_countrate  # also measure singles alongside the stream?

    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        import TimeTagger as TT  # lazy import of the vendor library

        tagger = device._require()  # ensure connected; get the raw hardware handle
        channels = list(params["general"]["timetagging"]["channels"])  # which channels to record
        ttbin_path = str(savepath.with_suffix(".ttbin"))               # output filename

        sync = TT.SynchronizedMeasurements(tagger)             # group so things start together
        fw = TT.FileWriter(sync.getTagger(), ttbin_path, channels)  # writes tags to .ttbin
        if self.max_file_size_mb > 0:
            # Convert MB to bytes and tell the writer to roll over into subfiles.
            fw.setMaxFileSize(int(self.max_file_size_mb * 1024 * 1024))

        countrate = None
        if self.record_countrate:
            # A Countrate runs in parallel to report singles, without any correlation math.
            countrate = TT.Countrate(sync.getTagger(), channels)

        logger.info("Recording raw tags -> %s (%.1f s)", ttbin_path, duration_ps * 1e-12)
        sync.startFor(int(duration_ps))  # record for the requested number of picoseconds
        sync.waitUntilFinished()          # block until that time elapses

        # Assemble a small result dict describing what was written.
        result = {
            "recorder": self.name,
            "file": ttbin_path,
            "duration_ps": int(duration_ps),
            "total_events": int(fw.getTotalEvents()),  # how many tags were written
            "total_size": int(fw.getTotalSize()),      # file size in bytes
        }
        if countrate is not None:
            counts = countrate.getCountsTotal()  # total counts per channel
            rates = countrate.getData()          # counts/s per channel
            # Build {channel_str: value} dicts, keeping channel order via enumerate.
            result["counts"] = {str(ch): int(counts[i]) for i, ch in enumerate(channels)}
            result["countrates"] = {str(ch): float(rates[i]) for i, ch in enumerate(channels)}

        del fw, countrate, sync  # release the measurement objects (stops the FileWriter)

        # Save a tiny sidecar pickle (parameters + counts) next to the .ttbin.
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
        self.config = config                              # keep the static configuration
        self.recorder = recorder or RawTimeTagRecorder()  # default to raw .ttbin recording
        self.stages = stages                              # optional rotation-stage controller
        self.stop_condition = stop_condition              # optional early-stop test

        self.time_hms, self.date = get_time_date()        # timestamp for this run
        # Build a unique run folder name like "2026-06-19_11-41-05_raw".
        self.save_dir = Path(config.base_dir) / f"{self.date}_{self.time_hms}_{config.experiment_type}"
        self.save_dir.mkdir(parents=True, exist_ok=True)  # create it (and parents) if missing
        self.logger = setup_logger(self.save_dir, name="acquisition")  # file+console logger

        # Use the provided device or build one from the config (not connected yet).
        self.device = device or TimeTaggerDevice(
            serial=config.tt_serial, resolution=config.timetagging.tt_mode,
            logger=self.logger)
        # Pre-compute the shared "general" parameter block used by every acquisition.
        self._general_params = self.config.general_params(self.date, self.time_hms, self.save_dir)

    # -- setup / teardown -----------------------------------------------------

    def setup(self) -> "Acquisition":
        """Connect the tagger, validate channels and apply the configuration."""
        self.logger.info("=" * 20 + " ACQUISITION SETUP " + "=" * 20)
        # Save the human-readable parameter file at the top of the run folder.
        save_json(self._general_params, self.save_dir / "general_parameters.json")

        if not self.device.is_connected:  # connect the tagger if not already
            self.device.connect()

        tt = self.config.timetagging       # shorthand for the tagger settings
        if not self.device.validate_channels(tt.channels):  # make sure channels exist
            raise RuntimeError("Requested channels are not all available; aborting.")

        # Apply the hardware configuration in order.
        self.device.set_trigger_levels(tt.channels, tt.trigger_levels_v)
        if any(tt.deadtime_ps):                       # only if any dead time is non-zero
            self.device.set_dead_times(tt.channels, tt.deadtime_ps)
        if any(tt.delays_ps):                         # only if any delay is non-zero
            self.device.set_input_delays(tt.channels, tt.delays_ps)
        if tt.trigger and tt.filter:                  # only if a conditional filter is configured
            self.device.set_conditional_filter(tt.trigger, tt.filter)
        self.logger.info("Recorder: %s", self.recorder.name)
        return self  # allow chaining: Acquisition(...).setup()

    def teardown(self):
        """Free the tagger and disconnect any stages."""
        self.device.free()              # release the TimeTagger
        if self.stages is not None:     # if we were given stages, release them too
            self.stages.disconnect_all()
        self.logger.info("Acquisition teardown complete.")

    # -- optional characterisation -------------------------------------------

    def characterize(self, label: str, duration_s: float = 5.0,
                     prompt: Optional[str] = None) -> dict:
        """Measure per-channel singles count rate (e.g. background vs signal).

        If ``prompt`` is given, wait for the user (cover/uncover the signal)
        before measuring. Returns the {channel: counts/s} mapping and logs it.
        """
        if prompt:               # e.g. "Cover the signal, then press Enter..."
            input(prompt)        # pause until the user presses Enter
        self.logger.info("Characterising '%s' for %.0f s ...", label, duration_s)
        # Use the device's simple count-rate probe over the requested duration.
        rates = self.device.measure_countrate(self.config.timetagging.channels, duration_s)
        for ch, r in rates.items():  # log each channel's measured rate
            self.logger.info("  %s ch %d: %.0f counts/s", label, ch, r)
        return rates

    # -- single point ---------------------------------------------------------

    def _experiment_params(self, duration_ps: int, savepath: Path,
                           point: ScanPoint) -> dict:
        # Build the per-acquisition "experimental" block of metadata.
        return {
            "type": self.config.experiment_type,
            "duration": int(duration_ps),       # acquisition length in picoseconds
            "savepath": str(savepath),
            "laser_power": point.power_mw,
            "rotation_stage": point.angle_deg,
            **point.extra,                       # splice in any extra per-point fields
        }

    def _move_stage_for(self, point: ScanPoint):
        # Only move if this point specifies an angle AND we actually have stages.
        if point.angle_deg is None or self.stages is None:
            return
        self.stages.move_to(point.angle_deg, point.stage_id)  # rotate to the requested angle
        self.logger.info("Stage %s -> %.4f deg", point.stage_id, point.angle_deg)

    def run_point(self, point: ScanPoint, duration_s: float,
                  file_stem: Optional[str] = None) -> dict:
        """Acquire a single fixed-duration measurement at ``point``."""
        self._move_stage_for(point)             # move the stage if needed
        duration_ps = int(duration_s * 1e12)    # seconds -> picoseconds (the tagger's unit)
        stem = file_stem or self._default_stem(point)  # filename stem (given or auto)
        savepath = self.save_dir / stem                 # full path (without extension)
        # Bundle the shared "general" block with this acquisition's "experimental" block.
        params = {"general": self._general_params,
                  "experimental": self._experiment_params(duration_ps, savepath, point)}
        self.logger.info("Single acquisition '%s' (%.2f min)", stem, duration_s / 60)
        # Delegate the actual recording to whichever recorder is plugged in.
        return self.recorder.record(self.device, duration_ps, savepath,
                                    params=params, logger=self.logger)

    # -- chunked --------------------------------------------------------------

    def run_chunked(self, point: ScanPoint, file_stem: Optional[str] = None) -> dict:
        """Acquire in chunks, merging into a running total until the stop
        condition is met or ``max_chunks`` is reached."""
        self._move_stage_for(point)                 # move the stage once before chunking
        ch = self.config.chunking                   # shorthand for chunking settings
        chunk_ps = int(ch.chunk_minutes * 60e12)    # chunk length in picoseconds
        stem = file_stem or self._default_stem(point)

        merged_dir = self.save_dir / "MERGED"       # subfolder for the running-total file
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_fp = merged_dir / f"{stem}_MERGED.pkl"  # the single merged-data file (overwritten)

        self.logger.info("Chunked acquisition '%s' (chunk=%.2f min, max=%d).",
                         stem, ch.chunk_minutes, ch.max_chunks)

        # Inner helper: wrap the running total `merged` with parameters whose
        # duration equals the TOTAL time recorded so far (n_done chunks).
        def _merged_payload(n_done: int) -> dict:
            """Wrap the running total with parameters whose duration reflects
            the *total* accumulated acquisition (n_done chunks), so downstream
            normalisation (pulses = duration * rep_rate) stays correct."""
            mp_params = copy.deepcopy(self._general_params)  # independent copy of metadata
            exp = self._experiment_params(chunk_ps * n_done, merged_fp, point)  # total duration
            return {"Parameters": {"general": mp_params, "experimental": exp},
                    "data": merged}

        merged: Optional[dict] = None  # the running total; None until the first chunk
        for i in range(ch.max_chunks):  # loop up to the safety cap
            chunk_stem = f"{stem}_chunk{i}"          # per-chunk filename stem
            savepath = self.save_dir / chunk_stem
            # Each individual chunk's params use the single-chunk duration.
            params = {"general": self._general_params,
                      "experimental": self._experiment_params(chunk_ps, savepath, point)}
            self.logger.info("Chunk %d/%d | elapsed %.1f min",
                             i + 1, ch.max_chunks, (i + 1) * ch.chunk_minutes)

            # Record this chunk and fold its result into the running total.
            result = self.recorder.record(self.device, chunk_ps, savepath,
                                          params=params, logger=self.logger)
            merged = self.recorder.merge(merged, result)

            # Periodically save the merged total and (optionally) test the stop condition.
            if (i + 1) % ch.check_every_n_chunks == 0:
                save_pickle(_merged_payload(i + 1), merged_fp)
                self.logger.info("Updated merged total -> %s", merged_fp)

                if self.stop_condition is not None:
                    stop, reason = self.stop_condition(merged)  # ask: enough data yet?
                    self.logger.info("Stop check: %s", reason)
                    if stop and ch.stop_when_reached:
                        self.logger.info(">>> STOP CONDITION REACHED - ending early <<<")
                        save_pickle(_merged_payload(i + 1), merged_fp)  # final save
                        self.logger.info("Final merged data -> %s", merged_fp)
                        return merged  # leave early; we have what we need

        # Reached the chunk cap without the stop condition firing: save and return.
        save_pickle(_merged_payload(ch.max_chunks), merged_fp)
        self.logger.info("Final merged data -> %s", merged_fp)
        return merged

    # -- scan -----------------------------------------------------------------

    def run_scan(self, points: Sequence[ScanPoint]) -> None:
        """Run every ``point`` (single or chunked per config) ``repeats`` times."""
        for num_repeat in range(self.config.repeats):  # outer loop: whole-scan repeats
            for point in points:                        # inner loop: each setting
                t0 = datetime.now()                     # remember when this point started
                stem = self._default_stem(point, num_repeat)  # unique filename per point+repeat
                if self.config.chunking.enabled:        # chunked or single, per config
                    self.run_chunked(point, file_stem=stem)
                else:
                    self.run_point(point, duration_s=self._default_single_duration_s(),
                                   file_stem=stem)
                dt_min = (datetime.now() - t0).total_seconds() / 60  # how long it took
                self.logger.info("Point '%s' completed in %.2f min", stem, dt_min)

    # -- helpers --------------------------------------------------------------

    def _default_single_duration_s(self) -> float:
        # When chunking is off, use one chunk's worth as the single duration.
        return self.config.chunking.chunk_minutes * 60.0

    def _default_stem(self, point: ScanPoint, num_repeat: int = 0) -> str:
        # Build a filename stem: prefer the point's label, else timestamp + power.
        if point.label:
            base = point.label
        else:
            # `:g` formats the number compactly (42.0 -> "42"); "run" if no power given.
            p = f"{point.power_mw:g}mW" if point.power_mw is not None else "run"
            base = f"{self.time_hms}_{self.date}_{p}"
        return f"{base}_num{num_repeat}"  # append the repeat index so names stay unique

    def __enter__(self) -> "Acquisition":
        return self.setup()  # `with Acquisition(...) as acq:` runs setup() automatically

    def __exit__(self, exc_type, exc, tb):
        self.teardown()  # ...and always tears down hardware on exit
        return False
