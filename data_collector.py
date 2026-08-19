import asyncio
import threading
import time
import csv
import os
import tkinter as tk
from tkinter import ttk
from collections import deque
from bleak import BleakClient, BleakScanner

# ---------------------------
# CONFIG
# ---------------------------
DEFAULT_BLE_ADDRESS = "E4:B3:23:F8:3B:5A"
CHARACTERISTIC_UUID = "0000FEF4-0000-1000-8000-00805F9B34FB"

FLUSH_EVERY_N_ROWS = 25
FLUSH_EVERY_SECONDS = 0.5
WARMUP_SAMPLES_TO_SKIP = 5

COLUMNS = [
    "t_ms",
    "Accel_X", "Accel_Y", "Accel_Z",
    "Gyro_X", "Gyro_Y", "Gyro_Z",
    "label",
    "session_id",
]

# ---------------------------
# GLOBAL STATE
# ---------------------------
ble_client = None
ble_thread = None
ble_loop = None
collecting = False

csv_filename = ""
buffer_rows = []
buffer_lock = threading.Lock()

rows_written = 0
last_flush_time = 0.0
warmup_left = 0

_csv_file = None
_csv_writer = None

# Device dropdown mapping
device_map = {}
scanning = False

# Live display state
latest_values_lock = threading.Lock()
latest_values = None               # tuple(ax,ay,az,gx,gy,gz)
total_rows_captured = 0
recv_times = deque(maxlen=80)      # timestamps of received packets


# ---------------------------
# CSV / BUFFERING
# ---------------------------
def open_csv_for_append(path: str):
    global _csv_file, _csv_writer
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    _csv_file = open(path, "a", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=COLUMNS)

    if not file_exists:
        _csv_writer.writeheader()
        _csv_file.flush()


def close_csv():
    global _csv_file, _csv_writer
    try:
        if _csv_file:
            _csv_file.flush()
            _csv_file.close()
    finally:
        _csv_file = None
        _csv_writer = None


def flush_buffer(force: bool = False):
    global buffer_rows, rows_written, last_flush_time

    if _csv_writer is None:
        return

    now = time.time()
    with buffer_lock:
        if not buffer_rows:
            return

        if (not force and
            len(buffer_rows) < FLUSH_EVERY_N_ROWS and
            (now - last_flush_time) < FLUSH_EVERY_SECONDS):
            return

        rows_to_write = buffer_rows
        buffer_rows = []

    for r in rows_to_write:
        _csv_writer.writerow(r)

    rows_written += len(rows_to_write)
    last_flush_time = now
    _csv_file.flush()


# ---------------------------
# THREAD-SAFE UI HELPERS
# ---------------------------
def set_status(msg: str):
    def _update():
        status_var.set(msg)
    root.after(0, _update)


def set_scan_button(enabled: bool):
    def _update():
        scan_btn.config(state=("normal" if enabled else "disabled"))
    root.after(0, _update)


def set_record_buttons(is_recording: bool):
    def _update():
        start_btn.config(state=("disabled" if is_recording else "normal"))
        stop_btn.config(state=("normal" if is_recording else "disabled"))
    root.after(0, _update)


def update_device_dropdown(options):
    def _update():
        device_combo["values"] = options
        if options:
            device_combo.current(0)
    root.after(0, _update)


def get_selected_address():
    selected = device_choice_var.get().strip()
    if selected in device_map:
        return device_map[selected]

    manual = address_var.get().strip()
    if manual:
        return manual

    return DEFAULT_BLE_ADDRESS


# ---------------------------
# BLE NOTIFICATIONS
# ---------------------------
def notification_handler(sender, data):
    """Keep this super light: parse -> buffer -> update live values."""
    global warmup_left, total_rows_captured, latest_values

    try:
        parts = data.decode(errors="ignore").strip().split(",")
        if len(parts) < 6:
            return

        values = list(map(int, parts[:6]))
        if warmup_left > 0:
            warmup_left -= 1
            return

        now = time.time()

        # Live stats
        with latest_values_lock:
            total_rows_captured += 1
            recv_times.append(now)
            latest_values = tuple(values)

        # CSV row
        row = {
            "t_ms": int(now * 1000),
            "Accel_X": values[0],
            "Accel_Y": values[1],
            "Accel_Z": values[2],
            "Gyro_X":  values[3],
            "Gyro_Y":  values[4],
            "Gyro_Z":  values[5],
            "label": gesture_var.get().strip(),
            "session_id": session_id_var.get().strip(),
        }

        with buffer_lock:
            buffer_rows.append(row)

    except Exception as e:
        print("Decode error:", e)


async def ble_task():
    """
    Windows BLE sometimes throws:
      Protocol Error 0x11: Insufficient Resource
    when stopping notifications quickly.
    We ignore stop_notify/disconnect errors and still exit cleanly.
    """
    global ble_client, collecting

    address = get_selected_address()
    if not address:
        set_status("No device selected. Click Scan and choose a device.")
        collecting = False
        set_record_buttons(False)
        return

    notified = False

    try:
        async with BleakClient(address) as client:
            ble_client = client
            set_status(f"Connected: {address}. Receiving data...")

            try:
                await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                notified = True
            except Exception as e:
                set_status(f"Start notify failed: {e}")
                collecting = False
                set_record_buttons(False)
                return

            while collecting:
                flush_buffer(force=False)
                await asyncio.sleep(0.05)

            # small settle time (helps on Windows)
            await asyncio.sleep(0.15)

            if notified:
                try:
                    await client.stop_notify(CHARACTERISTIC_UUID)
                except Exception as e:
                    print("Ignoring stop_notify error:", e)

            try:
                await client.disconnect()
            except Exception as e:
                print("Ignoring disconnect error:", e)

            set_status("Stopped. BLE Disconnected.")

    except Exception as e:
        set_status(f"BLE Error: {e}")
        print("BLE Error:", e)
    finally:
        ble_client = None
        set_record_buttons(False)


def run_ble_loop():
    global ble_loop
    ble_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ble_loop)
    ble_loop.run_until_complete(ble_task())


# ---------------------------
# BLE SCAN
# ---------------------------
def scan_devices():
    global scanning
    if scanning:
        return

    scanning = True
    set_scan_button(False)
    set_status("Scanning for BLE devices... (please wait)")

    def _scan_thread():
        global device_map, scanning
        try:
            devices = asyncio.run(BleakScanner.discover(timeout=6.0))
            temp_map = {}
            options = []

            for d in devices:
                name = (d.name or "Unknown").strip()
                addr = d.address
                label = f"{name} ({addr})"
                temp_map[label] = addr
                options.append(label)

            options.sort(key=lambda x: x.lower())
            device_map = temp_map

            if options:
                update_device_dropdown(options)
                set_status(f"Found {len(options)} device(s). Select one and press F5 to start.")
            else:
                set_status("No BLE devices found. Make sure device is ON & advertising.")

        except Exception as e:
            set_status(f"Scan failed: {e}")
        finally:
            scanning = False
            set_scan_button(True)

    threading.Thread(target=_scan_thread, daemon=True).start()


# ---------------------------
# START / STOP COLLECTION
# ---------------------------
def build_save_path(gesture: str, sess: str, file_no: str) -> str:
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    return os.path.join(
        desktop_path,
        "IMU_Gesture_Data",
        gesture,
        sess,
        f"sample{file_no}.csv"
    )


def start_ble_collection(event=None):
    global collecting, ble_thread, csv_filename, rows_written, last_flush_time, warmup_left
    global total_rows_captured, latest_values

    if collecting:
        return "break"

    gesture = gesture_var.get().strip()
    file_no = file_no_var.get().strip()
    sess = session_id_var.get().strip()

    if not gesture:
        set_status("Enter a valid gesture (e.g., up/down/left/right).")
        return "break"
    if not file_no.isdigit():
        set_status("Enter a valid file number (digits).")
        return "break"
    if not sess:
        set_status("Enter a session_id (e.g., day1_me, person2, etc.).")
        return "break"

    address = get_selected_address()
    if not address:
        set_status("No device selected. Click Scan and choose a device.")
        return "break"

    csv_filename = build_save_path(gesture, sess, file_no)

    # Reset counters and open CSV
    rows_written = 0
    last_flush_time = 0.0
    warmup_left = WARMUP_SAMPLES_TO_SKIP

    with latest_values_lock:
        total_rows_captured = 0
        latest_values = None
        recv_times.clear()

    close_csv()
    open_csv_for_append(csv_filename)

    collecting = True
    set_record_buttons(True)
    set_status(f"RECORDING (F6 to stop) | '{gesture}' | session '{sess}' | sample {file_no} | {address}\nSaved to: {csv_filename}")

    ble_thread = threading.Thread(target=run_ble_loop, daemon=True)
    ble_thread.start()

    return "break"


def stop_ble_collection(event=None):
    global collecting

    if not collecting:
        return "break"

    collecting = False

    # Ensure data is flushed even if BLE stop_notify fails
    flush_buffer(force=True)
    close_csv()

    # Auto increment file number
    file_no = file_no_var.get().strip()
    if file_no.isdigit():
        file_no_var.set(str(int(file_no) + 1))

    set_status(f"Stopped. Saved: {csv_filename}")
    # Buttons will be re-enabled in ble_task() finally as well
    set_record_buttons(False)

    return "break"


# ---------------------------
# LIVE DISPLAY UPDATER (GUI LOOP)
# ---------------------------
def update_live_panel():
    """Update the UI with latest values and stats."""
    if collecting:
        with latest_values_lock:
            count = total_rows_captured
            lv = latest_values
            times = list(recv_times)

        # packets/sec estimate
        pps = 0.0
        if len(times) >= 2:
            dt = times[-1] - times[0]
            if dt > 0:
                pps = (len(times) - 1) / dt

        samples_var.set(str(count))
        pps_var.set(f"{pps:.1f}")

        if lv is None:
            last_row_var.set("Waiting for packets...")
        else:
            ax, ay, az, gx, gy, gz = lv
            last_row_var.set(
                f"Ax:{ax:>6}  Ay:{ay:>6}  Az:{az:>6}   |   "
                f"Gx:{gx:>6}  Gy:{gy:>6}  Gz:{gz:>6}"
            )
        rec_var.set("REC ●")
    else:
        rec_var.set("IDLE")
    root.after(150, update_live_panel)


# ---------------------------
# GUI
# ---------------------------
root = tk.Tk()
root.title("BLE IMU Data Collector (F5 Start / F6 Stop)")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.resizable(False, False)

# Header with REC indicator
header = ttk.Frame(mainframe)
header.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 6))

ttk.Label(header, text="Controls: F5 = Start, F6 = Stop").grid(row=0, column=0, sticky=tk.W)
rec_var = tk.StringVar(value="IDLE")
ttk.Label(header, textvariable=rec_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky=tk.E, padx=(12, 0))
header.columnconfigure(0, weight=1)

# Device selection
ttk.Label(mainframe, text="BLE Device:").grid(row=1, column=0, sticky=tk.W, pady=(0, 3))
device_choice_var = tk.StringVar(value="")
device_combo = ttk.Combobox(mainframe, width=38, textvariable=device_choice_var, state="readonly")
device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 3))
scan_btn = ttk.Button(mainframe, text="Scan", command=scan_devices)
scan_btn.grid(row=1, column=2, padx=(8, 0), pady=(0, 3))

# Manual MAC fallback
ttk.Label(mainframe, text="Manual MAC (optional):").grid(row=2, column=0, sticky=tk.W)
address_var = tk.StringVar(value=DEFAULT_BLE_ADDRESS)
ttk.Entry(mainframe, width=22, textvariable=address_var).grid(row=2, column=1, sticky=tk.W)

# Gesture
ttk.Label(mainframe, text="Gesture Name:").grid(row=3, column=0, sticky=tk.W, pady=(8, 0))
gesture_var = tk.StringVar()
ttk.Entry(mainframe, width=22, textvariable=gesture_var).grid(row=3, column=1, sticky=tk.W, pady=(8, 0))

# Session
ttk.Label(mainframe, text="Session ID:").grid(row=4, column=0, sticky=tk.W)
session_id_var = tk.StringVar(value="day1_me")
ttk.Entry(mainframe, width=22, textvariable=session_id_var).grid(row=4, column=1, sticky=tk.W)

# File number
ttk.Label(mainframe, text="File Number:").grid(row=5, column=0, sticky=tk.W)
file_no_var = tk.StringVar(value="1")
ttk.Entry(mainframe, width=22, textvariable=file_no_var).grid(row=5, column=1, sticky=tk.W)

# Start/Stop buttons (also clickable)
start_btn = ttk.Button(mainframe, text="Start (F5)", command=start_ble_collection)
start_btn.grid(row=6, column=0, pady=10)
stop_btn = ttk.Button(mainframe, text="Stop (F6)", command=stop_ble_collection)
stop_btn.grid(row=6, column=1, pady=10, sticky=tk.W)
stop_btn.config(state="disabled")

# -------- Live panel --------
live = ttk.LabelFrame(mainframe, text="Live Recording Monitor", padding="10")
live.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(6, 6))

samples_var = tk.StringVar(value="0")
pps_var = tk.StringVar(value="0.0")
last_row_var = tk.StringVar(value="Not recording")

ttk.Label(live, text="Samples captured:").grid(row=0, column=0, sticky=tk.W)
ttk.Label(live, textvariable=samples_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(8, 0))

ttk.Label(live, text="Packets/sec:").grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
ttk.Label(live, textvariable=pps_var, font=("Segoe UI", 11, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(4, 0))

ttk.Label(live, text="Last row:").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
ttk.Label(live, textvariable=last_row_var).grid(row=2, column=1, sticky=tk.W, padx=(8, 0), pady=(6, 0))

# Status
status_var = tk.StringVar(value="Ready. Click Scan, select device, then press F5 to start.")
ttk.Label(mainframe, textvariable=status_var, wraplength=560).grid(row=8, column=0, columnspan=3, sticky=tk.W)

# Bind hotkeys
root.bind_all("<F5>", start_ble_collection)
root.bind_all("<F6>", stop_ble_collection)

def on_close():
    global collecting
    collecting = False
    try:
        flush_buffer(force=True)
        close_csv()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# Start live updater loop
root.after(150, update_live_panel)
root.mainloop()
