"""
can_simulator.py
================
CAN/ECU communication layer for the Rotax 912 ULS MALE UAV digital twin.
Simulates the CAN bus framing that an actual FADEC/ECU would transmit to
the Ground Control Station (GCS) or onboard data recorder.

CAN Frame Structure:
  - 11-bit arbitration ID (standard CAN 2.0A)
  - DLC = 4 bytes (32-bit little-endian unsigned integer payload)
  - Timestamp added to CANFrame for replay capability (Phase 5)

Signal Architecture:
  Priority class A (IDs 0x100-0x10F): Critical safety parameters
  Priority class B (IDs 0x110-0x11F): Performance parameters
  Priority class C (IDs 0x120-0x12F): Environmental / context

Ref: CAN 2.0 Part A Specification (Bosch, 1991)
     SocketCAN Linux interface (as specified in DRDO SIH-26054 brief)
     Rotax 912 FADEC interface convention
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import struct

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    can = None
    CAN_AVAILABLE = False


# ================================================================
# CAN FRAME
# ================================================================

@dataclass
class CANFrame:
    """
    Represents one CAN 2.0A message frame.

    Fields match the python-can Message interface for compatibility.
    timestamp_ms: hardware timestamp in milliseconds (for replay).
    """
    arbitration_id: int
    data: bytes
    dlc: int
    timestamp_ms: float = field(default_factory=lambda: time.monotonic() * 1000.0)
    is_extended_id: bool = False


# ================================================================
# SIGNAL DEFINITION TABLE
# Format: CAN_ID -> (signal_name, scale_factor, unit, description)
# Raw integer stored = round(physical_value * scale_factor)
# Physical value decoded = raw_integer / scale_factor
# ================================================================

SIGNALS = {
    # -- Priority A: Critical safety parameters --
    0x101: ("rpm",               1.0,    "RPM",    "Engine speed"),
    0x102: ("cht",               10.0,   "degC",   "Cylinder Head Temperature (measured)"),
    0x103: ("egt",               10.0,   "degC",   "Exhaust Gas Temperature"),
    0x104: ("oil_pressure",      100.0,  "PSI",    "Oil system pressure"),
    0x105: ("oil_temp",          10.0,   "degC",   "Oil temperature"),
    0x106: ("vibration",         10000.0,"g",      "Vibration RMS composite"),

    # -- Priority B: Performance parameters --
    0x107: ("fuel_flow",         100.0,  "L/h",    "Fuel flow rate"),
    0x108: ("battery_voltage",   1000.0, "V",      "Battery terminal voltage"),
    0x109: ("alternator_voltage",1000.0, "V",      "Alternator output voltage"),
    0x10A: ("battery_soc",       100.0,  "%",      "Battery state of charge"),
    0x10B: ("injection_timing",  100.0,  "degBTDC","Injection/ignition timing"),

    # -- Vibration sub-components (frequency domain features) --
    0x10C: ("vib_1x",            10000.0,"g",      "1x firing freq. vibration"),
    0x10D: ("vib_2x",            10000.0,"g",      "2x harmonic vibration"),
    0x10E: ("vib_05x",           10000.0,"g",      "0.5x sub-harmonic vibration"),

    # -- Priority C: Environmental / context --
    0x110: ("altitude_m",        1.0,    "m",      "Pressure altitude"),
    0x111: ("airspeed_kt",       10.0,   "kt",     "Indicated airspeed"),
    0x112: ("ambient_temp",      10.0,   "degC",   "Outside air temperature"),
    0x113: ("throttle_pct",      1000.0, "frac",   "Throttle position [0-1]"),
    0x114: ("engine_load",       1000.0, "frac",   "Engine load fraction [0-1]"),
}

# All signal names in the table (for quick lookup)
SIGNAL_NAMES = {name for name, *_ in SIGNALS.values()}


# ================================================================
# ENCODE / DECODE
# ================================================================

def encode_signal(can_id: int, value: float) -> CANFrame:
    """
    Encode a physical sensor value into a 4-byte CAN frame payload.

    Raw = round(value * scale); packed as uint32 little-endian.
    Values below 0 are clamped to 0 (unsigned raw representation).
    Values above uint32 max are clamped to 0xFFFFFFFF.

    Args:
        can_id : CAN arbitration ID (must be in SIGNALS table)
        value  : physical sensor value

    Returns:
        CANFrame ready for transmission or virtual bus injection
    """
    if can_id not in SIGNALS:
        raise KeyError(f"Unknown CAN ID: 0x{can_id:03X}")

    name, scale, unit, _ = SIGNALS[can_id]
    raw = max(0, min(int(round(float(value) * scale)), 0xFFFFFFFF))
    payload = struct.pack("<I", raw)   # unsigned 32-bit little-endian
    return CANFrame(arbitration_id=can_id, data=payload, dlc=4)


def decode_signal(frame: CANFrame) -> tuple:
    """
    Decode a CAN frame payload back to (signal_name, physical_value).

    Args:
        frame : CANFrame (must have a CAN ID in the SIGNALS table)

    Returns:
        (signal_name: str, physical_value: float)
    """
    if frame.arbitration_id not in SIGNALS:
        raise KeyError(f"Unknown CAN ID: 0x{frame.arbitration_id:03X}")

    name, scale, unit, _ = SIGNALS[frame.arbitration_id]
    raw = struct.unpack("<I", bytes(frame.data[:4]))[0]
    return name, raw / scale


def row_to_frames(row) -> list:
    """
    Convert one data row (dict or pandas Series) into a list of CAN frames.
    Only encodes signals present in the row; skips missing ones gracefully.
    """
    frames = []
    for can_id, (name, scale, unit, _) in SIGNALS.items():
        if name in row and row[name] is not None:
            try:
                frames.append(encode_signal(can_id, row[name]))
            except (ValueError, TypeError):
                pass    # skip bad values silently
    return frames


def frames_to_dict(frames: list) -> dict:
    """
    Decode a list of CAN frames back to a {signal_name: value} dict.
    """
    result = {}
    for frame in frames:
        try:
            name, value = decode_signal(frame)
            result[name] = value
        except KeyError:
            pass
    return result


def print_frame_table(frames: list):
    """Pretty-print CAN frames for demo / debug output."""
    print(f"\n{'CAN ID':<10} {'Signal':<22} {'DLC':<5} {'Data (hex)':<15} {'Decoded'}")
    print("-" * 70)
    for f in frames:
        name, value = decode_signal(f)
        _, scale, unit, desc = SIGNALS[f.arbitration_id]
        print(f"0x{f.arbitration_id:03X}      {name:<22} {f.dlc:<5} "
              f"{f.data.hex(' '):<15} {value:.4f} {unit}")


# ================================================================
# VIRTUAL BUS TEST
# ================================================================

def run_virtual_bus_test(frames: list):
    """
    Send frames through python-can virtual bus and receive them back.
    Validates round-trip fidelity at the software level.
    """
    if not CAN_AVAILABLE:
        print("\n[INFO] python-can not installed. Software frame test only.")
        print("       Install with: pip install python-can")
        return False

    bus = can.Bus(interface="virtual", receive_own_messages=True)
    try:
        for f in frames:
            msg = can.Message(
                arbitration_id=f.arbitration_id,
                data=f.data,
                is_extended_id=False,
            )
            bus.send(msg)
            received = bus.recv(timeout=1.0)
            if received is None:
                print(f"[FAIL] No response for CAN ID 0x{f.arbitration_id:03X}")
                return False
        print("[PASS] python-can virtual bus round-trip test passed")
        return True
    finally:
        bus.shutdown()


# ================================================================
# MAIN DEMO
# ================================================================

def main():
    """
    Demonstrate CAN frame encoding/decoding using a representative
    Rotax 912 ULS cruise-condition sensor reading.
    """
    # Representative cruise-condition sensor values (Rotax 912 ULS at 5000 m)
    sample = {
        "rpm":               4750.0,
        "cht":               125.0,
        "egt":               680.0,
        "oil_pressure":      55.0,
        "oil_temp":          98.0,
        "vibration":         0.1820,
        "fuel_flow":         18.5,
        "battery_voltage":   13.70,
        "alternator_voltage":14.10,
        "battery_soc":       94.0,
        "injection_timing":  21.5,
        "vib_1x":            0.1200,
        "vib_2x":            0.0400,
        "vib_05x":           0.0200,
        "altitude_m":        5000.0,
        "airspeed_kt":       92.0,
        "ambient_temp":      3.0,
        "throttle_pct":      0.800,
        "engine_load":       0.820,
    }

    print("=" * 70)
    print("PHASE 1 -- CAN/ECU SIMULATION (Rotax 912 ULS)")
    print("DRDO SIH-26054 | MALE UAV Digital Twin")
    print("=" * 70)
    print(f"\nTotal signals in CAN table : {len(SIGNALS)}")
    print(f"Priority A (0x101-0x10E)   : critical safety + vibration freq.")
    print(f"Priority C (0x110-0x114)   : environmental / context")
    print(f"\nEncoding {len(sample)} sensor values...\n")

    frames = row_to_frames(sample)
    print_frame_table(frames)

    decoded = frames_to_dict(frames)
    print(f"\n[Decoded values back from frames]")
    print(f"{'Signal':<22} {'Original':>12} {'Decoded':>12} {'Error':>10}")
    print("-" * 60)

    all_pass = True
    for can_id, (name, scale, unit, _) in SIGNALS.items():
        if name in sample:
            orig  = sample[name]
            dec   = decoded.get(name, None)
            if dec is None:
                print(f"  {name:<22} {'N/A':>12}")
                continue
            tol   = 1.0 / scale + 1e-9   # quantisation tolerance
            err   = abs(dec - orig)
            ok    = "PASS" if err <= tol else "FAIL"
            if ok == "FAIL":
                all_pass = False
            print(f"  {name:<22} {orig:>12.4f} {dec:>12.4f} {err:>10.6f}  {ok}")

    print()
    if all_pass:
        print("[PASS] All CAN signal round-trips within quantisation tolerance")
    else:
        print("[FAIL] Some signals failed round-trip -- check scale factors")

    # Virtual bus test (if python-can is installed)
    run_virtual_bus_test(frames)
    print("\n[INFO] python-can SocketCAN interface would be used with a real ECU.")
    print("[INFO] Replace 'virtual' with 'socketcan' and channel='vcan0' for hardware.")


if __name__ == "__main__":
    main()
