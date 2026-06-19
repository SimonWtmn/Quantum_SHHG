"""
=============================================================================
Hardware Abstraction Layer
=============================================================================

This module is strictly designed to talk to hardware.
It knows how to connect, configure, move devices, but it contains no acquisition logic and no physics. 

Devices
-------
* :class:`TimeTaggerDevice` - Swabian Instruments TimeTagger (Ultra / X / 20).
  Handles connection, resolution mode, per-channel trigger level / dead time /
  input delay, simple count-rate probes and clean teardown.
* :class:`RotationStage` - abstract base for a single motorised rotation stage,
  with two concrete implementations:
    - :class:`PRM1Stage`  : Thorlabs PRM1/MZ8 (K-Cube DC servo) via Kinesis .NET
      DLLs (``pythonnet``). Identified by a *string* serial number, e.g."27264707".
    - :class:`ELL14Stage` : Thorlabs ELL14 via the ``elliptec`` library over a
      serial port. Identified by an *integer* bus address 0..9.
* :class:`RotationStageController` - registry that connects to, addresses and
  releases several stages at once (mixed PRM1 + ELL14).

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
Date: 19/06/2026
"""

from __future__ import annotations

import os
import sys
import time
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import numpy as np


__all__ = [
    "TimeTaggerDevice",
    "RotationStage",
    "PRM1Stage",
    "ELL14Stage",
    "RotationStageController",
    "StageID",
]

StageID = Union[str, int]

_LOG = logging.getLogger("acquisition.hardware")


@contextmanager
def _suppress_stdout():
    """Silence chatty third-party libraries (e.g. elliptec) for one call."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old



# =============================================================================
# TimeTagger
# =============================================================================

class TimeTaggerDevice:
    """Connection + configuration wrapper for a Swabian Instruments TimeTagger.

    The object owns the underlying tagger handle and 
    exposes the configuration primitives the acquisition layer needs 
    (trigger levels, dead times, input delays, resolution mode) 
    plus lightweight count-rate probes used for background characterisation. 

    Parameters
    ----------
    serial : str, optional --> Serial number of the device to open. Empty string opens the first one.
    resolution : str --> Resolution mode name (``"Standard"``, ``"HighResA"``, ...). Mapped to ``TimeTagger.Resolution`` at connection time.
    logger : logging.Logger, optional --> Logger to report progress to (defaults to the module logger).
    """

    SUPPORTED_MODELS = ("Time Tagger 20", "Time Tagger Ultra", "Time Tagger X")

    def __init__(self, serial: str = "", resolution: str = "Standard", logger: Optional[logging.Logger] = None):
        self.serial = serial
        self.resolution = resolution
        self.log = logger or _LOG
        self.tagger = None
        self.model: Optional[str] = None


    # ----------------------- connection -----------------------

    @staticmethod
    def scan() -> list:
        """Return the list of serial numbers of all connected TimeTaggers."""
        import TimeTagger as TT
        return list(TT.scanTimeTagger())

    def connect(self) -> "TimeTaggerDevice":
        """Open the device in the requested resolution mode. Raises RuntimeError if no tagger can be opened or the model is unsupported."""
        import TimeTagger as TT
        found = TT.scanTimeTagger()
        self.log.info("Found %d connected Time Tagger(s)", len(found))
        resolution = getattr(TT.Resolution, self.resolution)
        try:
            self.tagger = TT.createTimeTagger(serial=self.serial, resolution=resolution)
        except Exception as exc: 
            raise RuntimeError(
                f"No Time Tagger found: {exc}"
            ) from exc

        self.serial = self.tagger.getSerial()
        self.model = self.tagger.getModel()
        if self.model not in self.SUPPORTED_MODELS:
            raise RuntimeError(f"Unsupported Time Tagger model: {self.model!r}")

        self.log.info("Time Tagger initialised (serial=%s, model=%s, resolution=%s).", self.serial, self.model, self.resolution)
        return self 

    @property
    def is_connected(self) -> bool:
        return self.tagger is not None

    def _require(self):
        if self.tagger is None:
            raise RuntimeError("TimeTagger is not connected; call connect() first.")
        return self.tagger



    # ----------------------- channel configuration -----------------------

    def available_channels(self, rising: bool = True) -> list:
        """Channels available in the current resolution mode (rising edges)."""
        import TimeTagger as TT
        tagger = self._require()
        if self.resolution == "Standard":
            edge = TT.ChannelEdge.Rising if rising else TT.ChannelEdge.Falling
        else:
            edge = TT.ChannelEdge.HighResRising if rising else TT.ChannelEdge.HighResFalling
        return list(tagger.getChannelList(edge))

    def validate_channels(self, channels: Sequence[int]) -> bool:
        """Return True iff every requested channel exists in this mode."""
        available = set(self.available_channels())
        ok = set(channels).issubset(available)
        if not ok:
            self.log.warning("Requested channels %s not all available (have %s).",
                             list(channels), sorted(available))
        return ok

    def set_trigger_levels(self, channels: Sequence[int], levels: Union[float, Sequence[float]]):
        """Set the trigger threshold (V) for the given channels.
        Parameters
        ----------
        channels : Sequence[int] --> Channels to set the trigger level for.
        levels : Union[float, Sequence[float]] --> Trigger level(s) in volts. Scalar applies to all channels, sequence applies one per channel.
        """
        tagger = self._require()
        if np.isscalar(levels):
            levels = [float(levels)] * len(channels)
        for ch, lvl in zip(channels, levels):
            tagger.setTriggerLevel(ch, float(lvl))
            self.log.info("Trigger level ch %d: %.3f V", ch, float(lvl))

    def set_dead_times(self, channels: Sequence[int], dead_times_ps: Sequence[float]):
        """Set the per-channel dead time in picoseconds."""
        tagger = self._require()
        for ch, dt in zip(channels, dead_times_ps):
            tagger.setDeadtime(ch, dt)
            self.log.info("Dead time ch %d: %s ps", ch, dt)

    def set_input_delays(self, channels: Sequence[int], delays_ps: Sequence[float]):
        """Set (and log the realised) per-channel input delay in picoseconds."""
        tagger = self._require()
        for ch, delay in zip(channels, delays_ps):
            tagger.setInputDelay(ch, delay)
            actual = tagger.getInputDelay(ch)
            self.log.info("Input delay ch %d: requested=%s ps, actual=%s ps", ch, delay, actual)

    def set_conditional_filter(self, trigger: Sequence[int], filtered: Sequence[int], hardware_delay_compensation: bool = True):
        """Enable the hardware conditional filter (keep ``filtered`` tags only when preceded by a ``trigger`` tag). No-op if either list is empty."""
        if not trigger or not filtered:
            return
        tagger = self._require()
        tagger.setConditionalFilter(trigger=list(trigger), filtered=list(filtered), hardwareDelayCompensation=hardware_delay_compensation)
        self.log.info("Conditional filter: trigger=%s, filtered=%s", list(trigger), list(filtered))

    def set_test_signal(self, channels: Sequence[int], enabled: bool = True):
        """Toggle the built-in periodic test signal on the given channels."""
        tagger = self._require()
        for ch in channels:
            tagger.setTestSignal(ch, enabled)
        self.log.info("Test signal %s on channels %s",
                      "enabled" if enabled else "disabled", list(channels))

    def get_trigger_levels(self, channels: Sequence[int]) -> Sequence[float]:
        """Get the trigger level (V) for the given channels."""
        tagger = self._require()
        return [tagger.getTriggerLevel(ch) for ch in channels]

    def get_dead_times(self, channels: Sequence[int]) -> Sequence[float]:
        """Get the dead time (ps) for the given channels."""
        tagger = self._require()
        return [tagger.getDeadtime(ch) for ch in channels]

    def get_input_delays(self, channels: Sequence[int]) -> Sequence[float]:
        """Get the input delay (ps) for the given channels."""
        tagger = self._require()
        return [tagger.getInputDelay(ch) for ch in channels]

    def get_conditional_filter(self, trigger: Sequence[int], filtered: Sequence[int]) -> bool:
        """Get the conditional filter (keep ``filtered`` tags only when preceded by a ``trigger`` tag)."""
        tagger = self._require()
        return tagger.getConditionalFilter(trigger=list(trigger), filtered=list(filtered))

    def get_test_signal(self, channels: Sequence[int]) -> bool:
        """Get the test signal (True/False) for the given channels."""
        tagger = self._require()
        return [tagger.getTestSignal(ch) for ch in channels]



    # ----------------------- simple probes -----------------------

    def measure_countrate(self, channels: Sequence[int], duration_s: float = 5.0) -> dict:
        """Average counts/s per channel over ``duration_s`` seconds.
        Parameters
        ----------
        channels : Sequence[int] --> Channels to measure the count rate for.
        duration_s : float --> Duration of the measurement in seconds.
        Returns
        -------
        dict --> Dictionary of channel numbers as keys and average counts per second as values.
        """
        import TimeTagger as TT
        tagger = self._require()

        sync = TT.SynchronizedMeasurements(tagger)
        counters = {ch: TT.Counter(sync.getTagger(), [ch], binwidth=int(1e12), n_values=int(max(duration_s, 1)) * 2) for ch in channels}
        sync.startFor(int(duration_s * 1e12))
        sync.waitUntilFinished()

        rates = {}
        for ch in channels:
            data = counters[ch].getData()
            total = float(np.sum(data))
            nonzero = int(np.count_nonzero(data))
            rates[ch] = total / max(nonzero, 1)
        del counters, sync  
        return rates



    # ----------------------- teardown / context manager -----------------------

    def free(self):
        """Release the TimeTagger handle."""
        if self.tagger is not None:
            import TimeTagger as TT
            TT.freeTimeTagger(self.tagger)
            self.log.info("Time Tagger freed (serial=%s).", self.serial)
            self.tagger = None

    def __enter__(self) -> "TimeTaggerDevice":
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.free()
        return False






# =============================================================================
# Rotation stages
# =============================================================================


class RotationStage(ABC):
    """Abstract single motorised rotation stage.

    Concrete subclasses implement the vendor-specific transport. Angles are
    always given in degrees and wrapped to ``[0, 360)`` by :meth:`move_to`.
    """

    def __init__(self, stage_id: StageID, logger: Optional[logging.Logger] = None):
        self.stage_id = stage_id
        self.log = logger or _LOG
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> "RotationStage":
        ...  # `...` (Ellipsis) is a placeholder body; abstract methods have no real code

    @abstractmethod
    def _move_to_impl(self, angle: float, extra_delay_s: float):
        ...  # vendor-specific "actually move" step, called by the shared move_to()

    @abstractmethod
    def position(self) -> Optional[float]:
        ...  # read the current angle

    @abstractmethod
    def home(self):
        ...  # send the stage to its reference/zero position

    @abstractmethod
    def disconnect(self):
        ...  # release the stage

    def move_to(self, angle: float, extra_delay_s: float = 0.1) -> "RotationStage":
        """Rotate to ``angle`` (deg, wrapped to 0-360) and block until settled."""
        if not self._connected:
            raise RuntimeError(f"Stage {self.stage_id} is not connected.")
        wrapped = float(angle) % 360.0
        self._move_to_impl(wrapped, extra_delay_s)
        self.log.info("Stage %s moved to %.4f deg", self.stage_id, wrapped)
        return self

    def __enter__(self) -> "RotationStage":
        if not self._connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()
        return False



class PRM1Stage(RotationStage):
    """Thorlabs PRM1/MZ8 rotation mount on a K-Cube DC servo, via Kinesis .NET.

    Requires ``pythonnet`` and the Thorlabs Kinesis DLLs (installed system-wide under ``C:\\Program Files\\Thorlabs\\Kinesis`` or pointed to by the ``KINESIS_DLL_PATH`` environment variable). Identified by a string serial.
    """

    _DEVICE_SETTINGS_NAME = "PRMTZ8"
    _MOVE_TIMEOUT_MS = 60000

    def __init__(self, serial: str, logger: Optional[logging.Logger] = None):
        super().__init__(serial, logger)  # run RotationStage.__init__ (sets id/log/_connected)
        self._device = None               # the Kinesis .NET device object (once connected)
        self._Decimal = None              # the .NET Decimal type (Kinesis wants Decimal angles)

    @staticmethod
    def _load_kinesis():
        """Import pythonnet and load the Kinesis assemblies; return the API tuple."""
        try:
            import clr  # provided by pythonnet; bridges Python <-> .NET DLLs
        except ImportError as exc:
            raise RuntimeError(
                "pythonnet is not installed. Run: pip install pythonnet"
            ) from exc
        if not hasattr(clr, "AddReference"):
            raise RuntimeError(
                "Wrong 'clr' package is installed (conflicts with pythonnet). "
                "Run: pip uninstall clr"
            )

        # Possible folders that contain the Thorlabs Kinesis DLLs.
        candidates = [
            os.environ.get("KINESIS_DLL_PATH"),    # user-provided override (may be None)
            r"C:\Program Files\Thorlabs\Kinesis",  # standard install location on Windows
        ]
        # Pick the first candidate that exists and is a real directory.
        dll_path = next((p for p in candidates if p and os.path.isdir(p)), None)
        if dll_path is None:
            raise RuntimeError(
                "Kinesis DLLs not found. Install Thorlabs Kinesis or set "
                "KINESIS_DLL_PATH to the folder containing the DLLs."
            )
        if dll_path not in sys.path:    # make the DLL folder importable...
            sys.path.append(dll_path)   # ...by adding it to Python's search path

        # Load the three .NET assemblies we need from the Kinesis SDK.
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
        # Now import the specific classes/types from those assemblies.
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
        from System import Decimal
        return DeviceManagerCLI, KCubeDCServo, Decimal  # hand them back to the caller

    @staticmethod
    def list_available() -> list:
        """Return serials of connected PRM1-class stages (serials starting '27')."""
        try:
            DeviceManagerCLI, _, _ = PRM1Stage._load_kinesis()  # only need the device manager
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Cannot list PRM1 stages: %s", exc)    # no DLLs / no pythonnet
            return []
        DeviceManagerCLI.BuildDeviceList()  # refresh the SDK's internal device list
        # Keep only serials that start with "27" (the PRM1/K-Cube family).
        return [str(sn) for sn in DeviceManagerCLI.GetDeviceList()
                if str(sn).startswith("27")]

    def connect(self) -> "PRM1Stage":
        DeviceManagerCLI, KCubeDCServo, Decimal = self._load_kinesis()  # load the .NET API
        self._Decimal = Decimal  # remember Decimal so move_to can convert angles later

        DeviceManagerCLI.BuildDeviceList()                       # refresh device list
        device = KCubeDCServo.CreateKCubeDCServo(self.stage_id)  # create the controller object
        if device is None:
            raise RuntimeError(f"Failed to create PRM1 device object for {self.stage_id}")

        device.Connect(self.stage_id)            # open the USB connection
        time.sleep(0.125)                        # tiny pause for the hardware to settle
        device.StartPolling(250)                 # poll device status every 250 ms
        time.sleep(0.125)
        device.EnableDevice()                    # energise the motor
        time.sleep(0.125)
        device.WaitForSettingsInitialized(10000)  # wait (<=10 s) for settings to load

        m_config = device.LoadMotorConfiguration(self.stage_id)  # load motor config
        m_config.DeviceSettingsName = self._DEVICE_SETTINGS_NAME  # set the mount profile
        m_config.UpdateCurrentConfiguration()                     # apply it
        device.SetSettings(device.MotorDeviceSettings, True, False)  # push settings to device

        self._device = device       # store the live device object
        self._connected = True      # mark as connected
        self.log.info("Connected to PRM1 stage %s", self.stage_id)
        return self

    def _move_to_impl(self, angle: float, extra_delay_s: float):
        # MoveTo needs a .NET Decimal; convert and give it the move timeout.
        self._device.MoveTo(self._Decimal(float(angle)), self._MOVE_TIMEOUT_MS)
        while self._device.Status.IsInMotion:  # keep checking until the motor stops
            time.sleep(0.05)                    # poll every 50 ms (don't busy-spin)
        if extra_delay_s > 0:                   # optional extra settle time
            time.sleep(extra_delay_s)

    def position(self) -> Optional[float]:
        if not self._connected:        # if not connected, there is no position to read
            return None
        # Position is a .NET Decimal; str() then float() converts it to a Python float.
        return float(str(self._device.Position))

    def home(self):
        if not self._connected:
            return
        self.log.info("Homing PRM1 stage %s ...", self.stage_id)
        self._device.Home(self._MOVE_TIMEOUT_MS)  # send to the mechanical reference point
        while self._device.Status.IsInMotion:     # wait until homing finishes
            time.sleep(0.1)

    def disconnect(self):
        if self._device is not None:               # only if we have a device to release
            try:
                self._device.StopPolling()         # stop the status polling thread
                self._device.Disconnect()          # close the USB connection
            except Exception as exc:  # noqa: BLE001
                # Never let a cleanup error crash the program; just warn.
                self.log.warning("Error disconnecting PRM1 %s: %s", self.stage_id, exc)
        self._device = None        # drop the reference
        self._connected = False    # mark disconnected
        self.log.info("Disconnected PRM1 stage %s", self.stage_id)


# ELL14Stage IS-A RotationStage too, but uses a totally different vendor library.
class ELL14Stage(RotationStage):
    """Thorlabs ELL14 resonant piezo rotation mount via the ``elliptec`` library.

    Several ELL14 share one serial-bus controller; this class connects (or
    reuses) the controller and binds to a single integer bus address ``0..9``.
    """

    # These are CLASS variables (shared by every ELL14Stage), because all ELL14
    # stages on one bus talk through a single serial controller.
    _controller = None                         # the shared elliptec.Controller (or None)
    _controller_port: Optional[str] = None     # which COM port the controller uses

    def __init__(self, address: int, port: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(address, logger)  # RotationStage.__init__ stores the address as id
        self.port = port                   # optional explicit COM port for the controller
        self._rotator = None               # the per-address elliptec.Rotator (once connected)

    @staticmethod
    def _load_elliptec():
        import elliptec                              # the ELL14 control library
        import serial.tools.list_ports as list_ports  # pyserial helper to list COM ports
        return elliptec, list_ports

    # `@classmethod` receives the class itself as `cls`; used here to touch the
    # shared class-level controller variables.
    @classmethod
    def _find_port(cls, list_ports) -> Optional[str]:
        # Collect COM ports whose description mentions a USB serial port.
        usb = [p.device for p in list_ports.comports()
               if "USB Serial Port" in (p.description or "")]
        if not usb:
            raise IOError("No USB Serial Port found for ELL14; check cables/drivers.")
        if len(usb) == 1:     # exactly one candidate -> use it automatically
            return usb[0]
        # With several ports we cannot blindly guess; ask the caller to specify.
        raise IOError(f"Multiple USB serial ports {usb}; pass an explicit `port`.")

    @classmethod
    def _ensure_controller(cls, port: Optional[str]):
        elliptec, list_ports = cls._load_elliptec()
        if cls._controller is not None:   # already connected once -> reuse it
            return cls._controller
        port = port or cls._find_port(list_ports)  # use given port or auto-detect one
        cls._controller = elliptec.Controller(port)  # open the shared serial controller
        cls._controller_port = port                   # remember which port it is on
        _LOG.info("Connected to ELL14 controller on %s", port)
        return cls._controller

    def connect(self) -> "ELL14Stage":
        elliptec, _ = self._load_elliptec()
        controller = self._ensure_controller(self.port)  # get/create the shared controller
        # Bind a Rotator object to this specific bus address on that controller.
        self._rotator = elliptec.Rotator(controller, address=str(self.stage_id))
        self._connected = True
        self.log.info("Connected to ELL14 stage at address %s", self.stage_id)
        return self

    def _move_to_impl(self, angle: float, extra_delay_s: float):
        with _suppress_stdout():            # hide the library's noisy prints
            self._rotator.set_angle(angle)  # command the rotation
        if extra_delay_s > 0:               # optional settle time
            time.sleep(extra_delay_s)

    def position(self) -> Optional[float]:
        if not self._connected:
            return None
        try:
            return float(self._rotator.get_angle())  # ask the rotator for its angle
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Could not read ELL14 %s position: %s", self.stage_id, exc)
            return None

    def home(self):
        if not self._connected:
            return
        self.log.info("Homing ELL14 address %s ...", self.stage_id)
        self._rotator.home()  # send this rotator to its reference position

    def disconnect(self):
        # Individual ELL14 stages just drop their reference; the shared
        # controller is closed by RotationStageController.disconnect_all().
        self._rotator = None
        self._connected = False
        self.log.info("Disconnected ELL14 address %s", self.stage_id)

    @classmethod
    def close_controller(cls):
        """Close the shared serial controller, if open."""
        ctrl = cls._controller
        if ctrl is None:   # nothing to close
            return
        try:
            # Different elliptec versions expose the serial port differently; try both.
            port = getattr(ctrl, "_port", None)            # newer: a `_port` attribute
            if port is not None and hasattr(port, "close"):
                port.close()
            elif hasattr(ctrl, "close"):                   # older: a `close()` method
                ctrl.close()
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Error closing ELL14 controller: %s", exc)
        finally:
            cls._controller = None        # always clear the shared references
            cls._controller_port = None


class RotationStageController:
    """Manage a collection of rotation stages addressed by their ID.

    A *string* ID is treated as a PRM1 serial, an *integer* ``0..9`` as an
    ELL14 bus address. This is the single entry point the acquisition layer
    uses to drive any number of mixed stages.

    Examples
    --------
    >>> stages = RotationStageController(["27264707", 2]).connect()
    >>> stages.move_to(45, "27264707")
    >>> stages.disconnect_all()
    """

    def __init__(self, stage_ids: Optional[Sequence[StageID]] = None,
                 logger: Optional[logging.Logger] = None):
        self.log = logger or _LOG
        # Maps each stage ID to its connected RotationStage object.
        self.stages: dict[StageID, RotationStage] = {}
        # The IDs we were asked to connect (copied into a list; [] if None given).
        self._requested = list(stage_ids) if stage_ids else []

    @staticmethod
    def _make_stage(stage_id: StageID, logger) -> RotationStage:
        # Factory: choose the right concrete class based on the ID's type.
        if isinstance(stage_id, str):                       # string -> PRM1 serial
            return PRM1Stage(stage_id, logger=logger)
        if isinstance(stage_id, int) and 0 <= stage_id <= 9:  # int 0..9 -> ELL14 address
            return ELL14Stage(stage_id, logger=logger)
        raise ValueError(f"Unrecognised stage id {stage_id!r} "
                         "(str -> PRM1 serial, int 0-9 -> ELL14 address).")

    def connect(self) -> "RotationStageController":
        """Connect to every requested stage (errors are logged, not fatal)."""
        for stage_id in self._requested:
            try:
                stage = self._make_stage(stage_id, self.log)  # build the right stage object
                stage.connect()                               # open it
                self.stages[stage_id] = stage                 # remember it on success
            except Exception as exc:  # noqa: BLE001
                # One bad stage should not stop the others from connecting.
                self.log.error("Could not connect stage %s: %s", stage_id, exc)
        self.log.info("Connected %d/%d rotation stage(s).",
                      len(self.stages), len(self._requested))
        return self

    def _resolve(self, stage_id: Optional[StageID]) -> RotationStage:
        # Turn an (optional) ID into a concrete connected stage, with helpful errors.
        if not self.stages:
            raise RuntimeError("No rotation stages connected.")
        if stage_id is None:                       # caller didn't say which stage
            if len(self.stages) > 1:               # ambiguous if more than one
                self.log.warning("Multiple stages connected; using first one.")
            return next(iter(self.stages.values()))  # just take the first connected stage
        if stage_id not in self.stages:
            raise KeyError(f"Stage {stage_id!r} is not connected.")
        return self.stages[stage_id]

    def move_to(self, angle: float, stage_id: Optional[StageID] = None,
                extra_delay_s: float = 0.1):
        # Find the requested stage and delegate the move to it.
        self._resolve(stage_id).move_to(angle, extra_delay_s)

    def position(self, stage_id: Optional[StageID] = None):
        if stage_id is None:  # no ID -> return a dict of every stage's position
            return {sid: s.position() for sid, s in self.stages.items()}
        return self._resolve(stage_id).position()  # otherwise just the one stage

    def home(self, stage_id: Optional[StageID] = None):
        if stage_id is None:           # no ID -> home all stages
            for s in self.stages.values():
                s.home()
        else:
            self._resolve(stage_id).home()  # otherwise home the one requested

    def disconnect_all(self):
        for stage in self.stages.values():  # disconnect each individual stage
            stage.disconnect()
        ELL14Stage.close_controller()       # then close the shared ELL14 serial controller
        self.stages.clear()                 # forget all stages
        self.log.info("All rotation stages disconnected.")

    def __enter__(self) -> "RotationStageController":
        return self.connect()  # `with RotationStageController(...) as stages:` connects

    def __exit__(self, exc_type, exc, tb):
        self.disconnect_all()  # ...and always disconnects on exit
        return False
