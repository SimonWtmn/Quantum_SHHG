"""

Thorlabs Rotation Stage Controller v2

Unified Python interface for controlling:
- Thorlabs PRM1/MZ8 rotation stages via Kinesis (string serial IDs like "27264707")
- Thorlabs ELL14 rotation stages via elliptec library (integer address IDs 0-9)

Requirements:
- For PRM1: pythonnet package (conda install pythonnet)
- For ELL14: elliptec and pyserial packages (pip install elliptec pyserial)

Usage:
    import thorlabs_rotation_stage_v2 as trs
    trs.connect_rotation_stages(["27264707", 2])  # Connect to PRM1 and ELL14
    trs.rotate_stage(10, "27264707")  # Rotate PRM1
    trs.rotate_stage(45, 2)  # Rotate ELL14
    trs.disconnect_all()
"""

import os
import sys
import time
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ============================================================================
# Type checking helpers
# ============================================================================

def _is_ell14_id(stage_id):
    """Check if ID is for ELL14 (integer 0-9)"""
    return isinstance(stage_id, int) and 0 <= stage_id <= 9

def _is_prm1_id(stage_id):
    """Check if ID is for PRM1 (string)"""
    return isinstance(stage_id, str)

# ============================================================================
# PRM1 Setup (Kinesis DLL)
# ============================================================================

def _load_kinesis_dlls():
    """Load Kinesis DLLs for PRM1 stages"""
    try:
        import clr
    except ImportError:
        print("WARNING: pythonnet not installed. PRM1 stages will not work.")
        print("Install with: conda install pythonnet")
        return None, None, None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_dll_path = os.path.join(script_dir, "dlls")
    system_dll_path = r"C:\Program Files\Thorlabs\Kinesis"

    if os.path.exists(system_dll_path):
        dll_path = system_dll_path
        print(f"Using system Kinesis DLLs from: {dll_path}")
    elif os.path.exists(local_dll_path):
        dll_path = local_dll_path
        print(f"Using local Kinesis DLLs from: {dll_path}")
    else:
        print("WARNING: Kinesis DLLs not found. PRM1 stages will not work.")
        return None, None, None

    sys.path.append(dll_path)
    try:
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
        clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")

        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
        from System import Decimal

        return DeviceManagerCLI, KCubeDCServo, Decimal
    except Exception as e:
        print(f"WARNING: Failed to load Kinesis DLLs: {e}")
        return None, None, None

# Try to load Kinesis DLLs
DeviceManagerCLI, KCubeDCServo, Decimal = _load_kinesis_dlls()

# ============================================================================
# ELL14 Setup (elliptec library)
# ============================================================================

def _load_elliptec():
    """Load elliptec library for ELL14 stages"""
    try:
        import elliptec
        import serial.tools.list_ports
        return elliptec, serial.tools.list_ports
    except ImportError:
        print("WARNING: elliptec not installed. ELL14 stages will not work.")
        print("Install with: pip install elliptec pyserial")
        return None, None

elliptec, serial_tools = _load_elliptec()

# ============================================================================
# Global state
# ============================================================================

_connected_prm1_stages = {}  # {serial_number: device_object}
_connected_ell14_stages = {}  # {address: rotator_object}
_ell14_controller = None  # Single controller for all ELL14 devices
_ell14_com_port = None  # COM port for ELL14 devices

# ============================================================================
# ELL14 Helper Functions
# ============================================================================

def _find_elliptec_port():
    """
    Auto-detect COM port for Elliptec devices.
    Probes USB Serial Ports to find responding Elliptec device.
    """
    if serial_tools is None or elliptec is None:
        raise ImportError("elliptec library not available")

    ports = list(serial_tools.comports())
    usb_ports = [p.device for p in ports if 'USB Serial Port' in p.description]

    if not usb_ports:
        raise IOError("No USB Serial Port found for ELL14; check cables and drivers")

    if len(usb_ports) == 1:
        print(f"Auto-detected single USB Serial Port: {usb_ports[0]}")
        return usb_ports[0]

    # Multiple ports - probe each one
    print(f"Detected {len(usb_ports)} USB Serial Ports. Probing for Elliptec device...")

    for port_name in usb_ports:
        try:
            print(f"  Trying {port_name}...", end=' ')
            controller = elliptec.Controller(port_name)

            # Try to query device info at common addresses
            for addr in ['0', '1', '2', '3']:
                try:
                    info = controller.get(address=addr)
                    if info:
                        print(f"✓ Found Elliptec device at address {addr}!")
                        return port_name
                except:
                    pass

            print("✗ No response")
        except Exception as e:
            print(f"✗ Failed")
            continue

    # Manual fallback
    print("\n⚠ No Elliptec device responded. Available ports:")
    for p in usb_ports:
        print(f"  • {p}")

    while True:
        choice = input("Enter port manually (e.g. COM3 or just 3), or 'q' to skip: ").strip()
        if choice.lower() == 'q':
            return None
        if choice.isdigit():
            choice = f"COM{choice}"
        if choice in usb_ports:
            return choice

def _connect_ell14_controller():
    """Connect to ELL14 controller if not already connected"""
    global _ell14_controller, _ell14_com_port

    if _ell14_controller is not None:
        return _ell14_controller  # Already connected

    if elliptec is None:
        raise ImportError("elliptec library not available")

    # Find COM port
    port = _find_elliptec_port()
    if port is None:
        raise IOError("Could not find or connect to ELL14 COM port")

    # Create controller
    _ell14_controller = elliptec.Controller(port)
    _ell14_com_port = port
    print(f"Connected to ELL14 controller on {port}")

    return _ell14_controller

def _connect_ell14_stage(address):
    """Connect to specific ELL14 stage by address"""
    global _connected_ell14_stages

    if address in _connected_ell14_stages:
        print(f"ELL14 address {address} already connected")
        return _connected_ell14_stages[address]

    # Ensure controller is connected
    controller = _connect_ell14_controller()

    # Create rotator for this address
    rotator = elliptec.Rotator(controller, address=str(address))

    # Verify device responds
    try:
        info = controller.get(address=str(address))
        if not info:
            print(f"WARNING: No response from ELL14 at address {address}")
    except Exception as e:
        print(f"WARNING: Could not verify ELL14 at address {address}: {e}")

    _connected_ell14_stages[address] = rotator
    # _home_ell14(address)
    print(f"Connected to ELL14 stage at address: {address}")

    return rotator

# ============================================================================
# PRM1 Helper Functions (original implementation)
# ============================================================================

def _connect_prm1_stage(serial_no):
    """Connect to specific PRM1 stage by serial number"""
    global _connected_prm1_stages

    if DeviceManagerCLI is None:
        raise ImportError("Kinesis DLLs not available")

    if serial_no in _connected_prm1_stages:
        print(f"PRM1 stage {serial_no} already connected")
        return _connected_prm1_stages[serial_no]

    try:
        device = KCubeDCServo.CreateKCubeDCServo(serial_no)
        if device is None:
            raise RuntimeError(f"Failed to create device object for {serial_no}")

        device.Connect(serial_no)
        time.sleep(0.125)

        device.StartPolling(250)
        time.sleep(0.125)
        device.EnableDevice()
        time.sleep(0.125)

        device.WaitForSettingsInitialized(10000)
        m_config = device.LoadMotorConfiguration(serial_no)
        m_config.DeviceSettingsName = "PRMTZ8"
        m_config.UpdateCurrentConfiguration()
        device.SetSettings(device.MotorDeviceSettings, True, False)

        _connected_prm1_stages[serial_no] = device
        print(f"Connected to PRM1 stage: {serial_no}")

        return device
    except Exception as e:
        print(f"ERROR: Failed to connect to PRM1 {serial_no}: {e}")
        raise

# ============================================================================
# Main Functions - Unified Interface
# ============================================================================

def connect_rotation_stages(serial_list=None):
    """
    Connect to rotation stages (PRM1 and/or ELL14)

    Parameters
    ----------
    serial_list : list, str, int, or None
        - List can contain strings (PRM1 serials) and/or integers (ELL14 addresses)
        - Single string (PRM1 serial) or integer (ELL14 address)
        - If None or empty, auto-discovers all available PRM1 stages

    Returns
    -------
    dict
        Dictionary of all connected stages {id: device_object}

    Examples
    --------
    >>> connect_rotation_stages(["27264707", 2])  # PRM1 + ELL14
    >>> connect_rotation_stages("27264707")  # Single PRM1
    >>> connect_rotation_stages(2)  # Single ELL14
    >>> connect_rotation_stages()  # Auto-discover PRM1
    """
    # Normalize input to list
    if serial_list is None:
        serial_list = []
    elif not isinstance(serial_list, list):
        serial_list = [serial_list]

    # Build PRM1 device list if needed
    if DeviceManagerCLI is not None:
        DeviceManagerCLI.BuildDeviceList()

    # Auto-discover PRM1 stages if list is empty
    if len(serial_list) == 0:
        if DeviceManagerCLI is not None:
            all_devices = DeviceManagerCLI.GetDeviceList()
            prm1_serials = [str(sn) for sn in all_devices if str(sn).startswith('27')]
            serial_list = prm1_serials
            if len(prm1_serials) > 0:
                print(f"Auto-discovered {len(prm1_serials)} PRM1 stage(s): {prm1_serials}")

    # Separate PRM1 and ELL14 IDs
    prm1_ids = [s for s in serial_list if _is_prm1_id(s)]
    ell14_ids = [s for s in serial_list if _is_ell14_id(s)]

    # Connect to PRM1 stages
    for serial_no in prm1_ids:
        try:
            _connect_prm1_stage(serial_no)
        except Exception as e:
            print(f"ERROR: Could not connect to PRM1 {serial_no}: {e}")

    # Connect to ELL14 stages
    for address in ell14_ids:
        try:
            _connect_ell14_stage(address)
        except Exception as e:
            print(f"ERROR: Could not connect to ELL14 address {address}: {e}")

    # Return combined dictionary
    all_stages = {**_connected_prm1_stages, **_connected_ell14_stages}

    if len(all_stages) > 0:
        print(f"\nTotal connected stages: {len(all_stages)} "
              f"(PRM1: {len(_connected_prm1_stages)}, ELL14: {len(_connected_ell14_stages)})")

    return all_stages

def rotate_stage(angle, serial=None, extra_delay=100):
    """
    Rotate stage to target angle and wait for completion

    Automatically wraps angles to 0-360° range.

    Parameters
    ----------
    angle : float
        Target angle in degrees (will be wrapped to 0-360°)
    serial : str or int, optional
        Stage ID (string for PRM1, integer for ELL14)
        If None, uses first available stage (with warning if multiple connected)
    extra_delay : int, optional
        Additional wait time in milliseconds after rotation (default: 100)
    """
    total_stages = len(_connected_prm1_stages) + len(_connected_ell14_stages)

    if total_stages == 0:
        print("ERROR: No stages connected. Call connect_rotation_stages() first.")
        return

    # Wrap angle to 0-360°
    wrapped_angle = angle % 360

    # Determine which stage to use
    if serial is None:
        all_ids = list(_connected_prm1_stages.keys()) + list(_connected_ell14_stages.keys())
        if total_stages > 1:
            serial = min(all_ids, key=lambda x: (isinstance(x, str), x))
            print(f"WARNING: Multiple stages connected. Using stage {serial}")
        else:
            serial = all_ids[0]

    # Route to appropriate handler
    if _is_prm1_id(serial):
        _rotate_prm1(wrapped_angle, serial, extra_delay)
    elif _is_ell14_id(serial):
        _rotate_ell14(wrapped_angle, serial, extra_delay)
    else:
        print(f"ERROR: Invalid stage ID: {serial}")

def _rotate_prm1(angle, serial, extra_delay):
    """Rotate PRM1 stage"""
    if serial not in _connected_prm1_stages:
        print(f"ERROR: PRM1 stage {serial} not connected")
        return

    device = _connected_prm1_stages[serial]

    try:
        device.MoveTo(Decimal(float(angle)), 60000)

        while device.Status.IsInMotion:
            time.sleep(0.05)

        if extra_delay > 0:
            time.sleep(extra_delay / 1000.0)
    except Exception as e:
        print(f"ERROR: Failed to rotate PRM1 {serial}: {e}")

def _rotate_ell14(angle, address, extra_delay):
    """Rotate ELL14 stage"""
    if address not in _connected_ell14_stages:
        print(f"ERROR: ELL14 address {address} not connected")
        return

    rotator = _connected_ell14_stages[address]

    try:
        with suppress_stdout():
            rotator.set_angle(angle)

        if extra_delay > 0:
            time.sleep(extra_delay / 1000.0)
    except Exception as e:
        print(f"ERROR: Failed to rotate ELL14 address {address}: {e}")

def home_stage(serial=None):
    """
    Home the rotation stage(s)

    Parameters
    ----------
    serial : str, int, or None
        Stage ID to home. If None, homes all connected stages.
    """
    total_stages = len(_connected_prm1_stages) + len(_connected_ell14_stages)

    if total_stages == 0:
        print("ERROR: No stages connected.")
        return

    if serial is None:
        # Home all stages
        for prm1_serial in _connected_prm1_stages.keys():
            _home_prm1(prm1_serial)
        for ell14_addr in _connected_ell14_stages.keys():
            _home_ell14(ell14_addr)
    elif _is_prm1_id(serial):
        _home_prm1(serial)
    elif _is_ell14_id(serial):
        _home_ell14(serial)
    else:
        print(f"ERROR: Invalid stage ID: {serial}")

def _home_prm1(serial):
    """Home PRM1 stage"""
    if serial not in _connected_prm1_stages:
        print(f"WARNING: PRM1 stage {serial} not connected, skipping")
        return

    device = _connected_prm1_stages[serial]

    try:
        print(f"Homing PRM1 stage {serial}...")
        device.Home(60000)

        while device.Status.IsInMotion:
            time.sleep(0.1)

        print(f"PRM1 stage {serial} homed successfully")
    except Exception as e:
        print(f"ERROR: Failed to home PRM1 {serial}: {e}")

def _home_ell14(address):
    """Home ELL14 stage"""
    if address not in _connected_ell14_stages:
        print(f"WARNING: ELL14 address {address} not connected, skipping")
        return

    rotator = _connected_ell14_stages[address]

    try:
        print(f"Homing ELL14 address {address}...")
        rotator.home()
        print(f"ELL14 address {address} homed successfully")
    except Exception as e:
        print(f"ERROR: Failed to home ELL14 address {address}: {e}")

def get_position(serial=None):
    """
    Get current position of rotation stage(s)

    Parameters
    ----------
    serial : str, int, or None
        Stage ID. If None, returns positions of all connected stages.

    Returns
    -------
    float or dict
        Current angle in degrees (single stage) or {id: angle} (all stages)
    """
    total_stages = len(_connected_prm1_stages) + len(_connected_ell14_stages)

    if total_stages == 0:
        print("ERROR: No stages connected.")
        return None

    if serial is None:
        # Get all positions
        positions = {}

        for prm1_serial in _connected_prm1_stages.keys():
            pos = _get_position_prm1(prm1_serial)
            if pos is not None:
                positions[prm1_serial] = pos

        for ell14_addr in _connected_ell14_stages.keys():
            pos = _get_position_ell14(ell14_addr)
            if pos is not None:
                positions[ell14_addr] = pos

        return positions
    elif _is_prm1_id(serial):
        return _get_position_prm1(serial)
    elif _is_ell14_id(serial):
        return _get_position_ell14(serial)
    else:
        print(f"ERROR: Invalid stage ID: {serial}")
        return None

def _get_position_prm1(serial):
    """Get position of PRM1 stage"""
    if serial not in _connected_prm1_stages:
        print(f"ERROR: PRM1 stage {serial} not connected")
        return None

    device = _connected_prm1_stages[serial]
    return float(str(device.Position))

def _get_position_ell14(address):
    """Get position of ELL14 stage"""
    if address not in _connected_ell14_stages:
        print(f"ERROR: ELL14 address {address} not connected")
        return None

    rotator = _connected_ell14_stages[address]
    try:
        return rotator.get_angle()
    except Exception as e:
        print(f"ERROR: Failed to get position of ELL14 address {address}: {e}")
        return None

def disconnect_all():
    """Disconnect all connected rotation stages (PRM1 and ELL14)"""
    global _connected_prm1_stages, _connected_ell14_stages
    global _ell14_controller, _ell14_com_port
    
    # Disconnect PRM1 stages
    for serial, device in _connected_prm1_stages.items():
        try:
            device.StopPolling()
            device.Disconnect()
            print(f"Disconnected PRM1 stage: {serial}")
        except Exception as e:
            print(f"ERROR disconnecting PRM1 {serial}: {e}")
    
    # Disconnect ELL14 stages (just clear references)
    for address in _connected_ell14_stages.keys():
        print(f"Disconnected ELL14 address: {address}")
    
    # Close ELL14 controller - ALWAYS if it exists
    if _ell14_controller is not None:
        try:
            # Close the underlying serial connection
            if hasattr(_ell14_controller, '_port') and _ell14_controller._port is not None:
                if hasattr(_ell14_controller._port, 'close'):
                    _ell14_controller._port.close()
                    print(f"Closed ELL14 controller serial port on {_ell14_com_port}")
            elif hasattr(_ell14_controller, 'close'):
                _ell14_controller.close()
                print(f"Closed ELL14 controller on {_ell14_com_port}")
            else:
                print(f"Cleared ELL14 controller reference for {_ell14_com_port}")
        except Exception as e:
            print(f"ERROR closing ELL14 controller: {e}")
    
    _connected_prm1_stages = {}
    _connected_ell14_stages = {}
    _ell14_controller = None
    _ell14_com_port = None
    
    print("All stages disconnected")


def home_all():
    """
    Home all connected rotation stages (convenience function)
    
    Equivalent to home_stage(None) - homes all PRM1 and ELL14 stages.
    """
    home_stage(None)

def list_available_stages():
    """
    List all available rotation stages

    Returns
    -------
    dict
        {'prm1': [serial_numbers], 'ell14': 'auto-detect needed'}
    """
    available = {'prm1': [], 'ell14': 'auto-detect on connect'}

    # List PRM1 stages
    if DeviceManagerCLI is not None:
        DeviceManagerCLI.BuildDeviceList()
        all_devices = DeviceManagerCLI.GetDeviceList()
        prm1_stages = [str(sn) for sn in all_devices if str(sn).startswith('27')]
        available['prm1'] = prm1_stages

        if len(prm1_stages) > 0:
            print(f"Available PRM1 stages: {prm1_stages}")
        else:
            print("No PRM1 stages found")
    else:
        print("Kinesis DLLs not available - cannot list PRM1 stages")

    # ELL14 stages require connection to detect
    if elliptec is not None:
        print("ELL14 stages: Auto-detected during connection")
    else:
        print("elliptec library not available - cannot use ELL14 stages")

    return available

# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Thorlabs Rotation Stage Controller v2 - Test")
    print("=" * 60)

    # Test 1: List available stages
    print("\n[Test 1] Listing available stages...")
    available = list_available_stages()

    # Test 2: Connect to specific stages
    print("\n[Test 2] Connecting to stages...")
    # Modify these IDs for your setup
    test_stages = [2]  # Example: ELL14 at address 2
    # test_stages = ["27264707"]  # Example: PRM1
    # test_stages = ["27264707", 2]  # Example: Both

    try:
        connect_rotation_stages(test_stages)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    if len(_connected_prm1_stages) + len(_connected_ell14_stages) == 0:
        print("\nNo stages connected. Exiting test.")
        sys.exit(0)
        disconnect_all()

    # Test 3: Get current positions
    print("\n[Test 3] Getting current positions...")
    positions = get_position()
    print(f"Current positions: {positions}")

    # Test 4: Rotate to various angles
    print("\n[Test 4] Rotating stages...")
    test_angles = [0, 45, 90]

    for angle in test_angles:
        print(f"\n  Rotating to {angle}°...")
        for stage_id in test_stages:
            rotate_stage(angle, stage_id)
        time.sleep(0.5)

        pos = get_position()
        print(f"  Current position(s): {pos}")

    # Test 5: Home stages
    print("\n[Test 5] Homing stages...")
    for stage_id in test_stages:
        home_stage(stage_id)

    # Test 6: Disconnect
    print("\n[Test 6] Disconnecting all stages...")
    disconnect_all()

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
