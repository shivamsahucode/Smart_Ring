import asyncio
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque

import numpy as np
import pandas as pd
import joblib
from bleak import BleakClient, BleakScanner
import pyautogui


pyautogui.FAILSAFE = False         
pyautogui.PAUSE = 0                

MOUSE_UPDATE_MS = 16               
MOVE_SCALE = 0.02                  
GYRO_SMOOTH = 0.70                 
EPS_GYRO = 8.0


IDLE_LOCK_WAIT_MS = 400         
LOCK_DURATION_MS = 400           

# ML Click Trigger
TAP_JERK_TRIGGER = 3000.0          
ML_CAPTURE_WINDOW_MS = 600         




# 🚨 IDLE THRESHOLDS (Hybrid Fallback)
ACTIVITY_GYRO_RMS_MIN = 15.0       
ACTIVITY_ACC_HP_RMS_MIN = 200.0    

# =========================
# CONFIG 
# =========================
GESTURE_MODEL_PATH = "gesture_rf.joblib"
CLICK_MODEL_PATH = "click_rf.joblib"    

CHARACTERISTIC_UUID = "0000FEF4-0000-1000-8000-00805F9B34FB"
DEFAULT_BLE_ADDRESS = ""
SENSOR_COLS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

RESAMPLE_DT_MS = 15
WINDOW_SAMPLES = 40                
HP_GRAVITY_KERNEL = 15             
BUFFER_SECONDS = 2.5
PREDICT_EVERY_MS = 100
STABLE_REQUIRED = 2
MIN_CONFIDENCE = 0.40

# =========================
# MODELS LOAD
# =========================
gesture_model = joblib.load(GESTURE_MODEL_PATH)
click_model = joblib.load(CLICK_MODEL_PATH)

# =========================
# GLOBAL STATE
# =========================
collecting = False
ble_thread = None
ble_loop = None
ble_client = None
device_map = {}
scanning = False

buf_lock = threading.Lock()
buf = deque()
fast_sensor_buf = deque(maxlen=4)

stable_label = "—"
last_pred_label = None
same_count = 0

# Mouse & ML Click state
mouse_active = False
prev_move = np.array([0.0, 0.0])
prev_acc_mag = 0.0

idle_time_ms = 0  
cursor_freeze_time = 0.0  

# ML Click Capture State
click_capture_active = False
click_capture_start_time = 0.0

# =========================
# FEATURES & MATH
# =========================
def high_pass_mag(acc_mag, kernel=15):
    if len(acc_mag) < kernel: return acc_mag - acc_mag.mean()
    k = np.ones(kernel) / kernel
    trend = np.convolve(acc_mag, k, mode="same")
    return acc_mag - trend

def resample_fixed_dt_from_buffer(rows, dt_ms=15):
    if len(rows) < 2: return None
    arr = np.array(rows, dtype=float)
    t = arr[:, 0]; t = t - t[0]
    _, uniq_idx = np.unique(t, return_index=True)
    t = t[uniq_idx]; x = arr[uniq_idx, 1:]
    if len(t) < 2: return None
    new_t = np.arange(0, t[-1] + 1, dt_ms)
    out = pd.DataFrame({"t_ms": new_t + arr[0, 0]})
    for i, col in enumerate(SENSOR_COLS): out[col] = np.interp(new_t, t, x[:, i])
    return out

def extract_last_window_features(df, window_samples=40):
    if df is None or len(df) < window_samples: return None
    x = df[SENSOR_COLS].to_numpy(dtype=float)
    acc = x[:, 0:3]; gyr = x[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    acc_hp = high_pass_mag(acc_mag, kernel=HP_GRAVITY_KERNEL)
    sig = np.column_stack([gyr, gyr_mag, acc_mag, acc_hp])  
    w = sig[-window_samples:]  
    means = w.mean(axis=0); stds = w.std(axis=0)
    mins = w.min(axis=0); maxs = w.max(axis=0)
    rng = maxs - mins; rms = np.sqrt(np.mean(w ** 2, axis=0))
    slope = w[-1] - w[0]
    return np.concatenate([means, stds, mins, maxs, rng, rms, slope]).reshape(1, -1)

def activity_score(df):
    if df is None or len(df) < WINDOW_SAMPLES: return 0.0, 0.0
    x = df[SENSOR_COLS].to_numpy(dtype=float)
    acc = x[:, 0:3]
    gyr = x[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    acc_hp = high_pass_mag(acc_mag, kernel=HP_GRAVITY_KERNEL)
    w_gyr = gyr[-WINDOW_SAMPLES:]
    w_hp = acc_hp[-WINDOW_SAMPLES:]
    gyro_rms = float(np.sqrt(np.mean(w_gyr ** 2)))
    acc_hp_rms = float(np.sqrt(np.mean(w_hp ** 2)))
    return gyro_rms, acc_hp_rms

def calculate_pointer_move(gx, gy, gz):
    global prev_move
    raw_move = np.array([gz, -gy])  
    move = GYRO_SMOOTH * prev_move + (1.0 - GYRO_SMOOTH) * raw_move
    prev_move = move
    speed = np.linalg.norm(move)
    if speed < EPS_GYRO: return 0, 0, speed
    return int(move[0] * MOVE_SCALE), int(move[1] * MOVE_SCALE), speed

# =========================
# BLE HANDLING
# =========================
def notification_handler(sender, data):
    try:
        parts = data.decode(errors="ignore").strip().split(",")
        if len(parts) < 6: return
        vals = list(map(float, parts[:6]))
        t_ms = int(time.time() * 1000)
        with buf_lock:
            buf.append((t_ms, *vals))
            cutoff = t_ms - int(BUFFER_SECONDS * 1000)
            while buf and buf[0][0] < cutoff: buf.popleft()
        fast_sensor_buf.append(vals)
    except: pass 

async def ble_task(address):
    global collecting, ble_client
    notified = False
    try:
        async with BleakClient(address) as client:
            ble_client = client
            set_status(f"Connected: {address}. Streaming...")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            notified = True
            while collecting: await asyncio.sleep(0.05)
            await asyncio.sleep(0.15)
            if notified:
                try: await client.stop_notify(CHARACTERISTIC_UUID)
                except: pass
            try: await client.disconnect()
            except: pass
            set_status("Disconnected.")
    except Exception as e:
        set_status(f"BLE Error: {e}")
    finally:
        ble_client = None
        stop_ui_state()

def run_ble_loop(address):
    global ble_loop
    ble_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ble_loop)
    ble_loop.run_until_complete(ble_task(address))

# =========================
# 🚀 FAST LOOP: MOUSE & ML CLICK CAPTURE (~60 FPS)
# =========================
def mouse_tick():
    global prev_acc_mag, cursor_freeze_time, prev_move
    global click_capture_active, click_capture_start_time

    if mouse_active and collecting and fast_sensor_buf:
        current_time = time.time() * 1000
        recent_data = np.array(fast_sensor_buf)
        
        latest_acc = recent_data[-1, 0:3] 
        curr_acc_mag = np.linalg.norm(latest_acc)
        jerk = abs(curr_acc_mag - prev_acc_mag)
        prev_acc_mag = curr_acc_mag
        
        jerk_var.set(f"{jerk:.1f}")

        # --- 1. START ML CLICK RECORDING ---
        if current_time <= cursor_freeze_time:
            if not click_capture_active and jerk > TAP_JERK_TRIGGER:
                click_capture_active = True
                click_capture_start_time = current_time
                prev_move = np.array([0.0, 0.0])
                cursor_freeze_time = current_time + ML_CAPTURE_WINDOW_MS + 100 # Extend shield
                print("Recording click data...")

        # --- 2. EXECUTE ML PREDICTION ---
        if click_capture_active and (current_time - click_capture_start_time >= ML_CAPTURE_WINDOW_MS):
            with buf_lock:
                rows = list(buf)
            df = resample_fixed_dt_from_buffer(rows, dt_ms=RESAMPLE_DT_MS)
            feat = extract_last_window_features(df, window_samples=WINDOW_SAMPLES)
            
            if feat is not None:
                ml_click_pred = click_model.predict(feat)[0].lower()
                print(f"ML CLICK PREDICTION: {ml_click_pred}")
                
                if ml_click_pred == "click":
                    pyautogui.click()
                elif ml_click_pred == "double_click":
                    pyautogui.doubleClick()

            click_capture_active = False 

        # --- 3. MOUSE MOVEMENT ---
        if current_time > cursor_freeze_time:
            gx, gy, gz = recent_data[:, 3:6].mean(axis=0)
            dx, dy, speed = calculate_pointer_move(gx, gy, gz)
            mouse_speed_var.set(f"{speed:.1f}")

            if dx != 0 or dy != 0:
                try: pyautogui.moveRel(dx, dy, duration=0)
                except: pass 
        else:
            mouse_speed_var.set("IDLE LOCK 🔒")

    root.after(MOUSE_UPDATE_MS, mouse_tick)

# =========================
# 🐢 SLOW LOOP: HYBRID IDLE GATING (10 FPS)
# =========================
def prediction_tick():
    global stable_label, last_pred_label, same_count
    global idle_time_ms, cursor_freeze_time

    if collecting:
        with buf_lock: rows = list(buf)
        df = resample_fixed_dt_from_buffer(rows, dt_ms=RESAMPLE_DT_MS)
        feat = extract_last_window_features(df, window_samples=WINDOW_SAMPLES)

        if feat is not None:
            # Gesture ML Prediction 
            pred = gesture_model.predict(feat)[0]
            if hasattr(gesture_model, "predict_proba"):
                proba = gesture_model.predict_proba(feat)[0]
                conf = float(np.max(proba))
            else: conf = 1.0

            if conf >= MIN_CONFIDENCE:
                if pred == last_pred_label: same_count += 1
                else: last_pred_label = pred; same_count = 1
            else: last_pred_label = None; same_count = 0

            if same_count >= STABLE_REQUIRED: stable_label = pred

            # --- 🔥 HYBRID AIM AND HOLD LOGIC ---
            gyro_rms, acc_hp_rms = activity_score(df)

            # Check Model
            model_says_idle = (str(stable_label).strip().upper() == "IDLE")
            # Check Sensors
            sensors_say_idle = (gyro_rms < ACTIVITY_GYRO_RMS_MIN and acc_hp_rms < ACTIVITY_ACC_HP_RMS_MIN)

            # If EITHER the model says we are idle, OR the raw math says we are idle...
            if model_says_idle or sensors_say_idle:
                idle_time_ms += PREDICT_EVERY_MS
                if idle_time_ms >= IDLE_LOCK_WAIT_MS:
                    # Timer hit! Freeze the cursor!
                    cursor_freeze_time = (time.time() * 1000) + LOCK_DURATION_MS 
            else:
                idle_time_ms = 0

            update_prediction_ui(stable_label, pred, conf, len(rows), gyro_rms, acc_hp_rms)

    root.after(PREDICT_EVERY_MS, prediction_tick)

# =========================
# GUI HELPERS
# =========================
def set_status(msg): root.after(0, lambda: status_var.set(msg))
def update_device_dropdown(options):
    def _u():
        device_combo["values"] = options
        if options: device_combo.current(0)
    root.after(0, _u)

def update_prediction_ui(stable, current, conf, nrows, gyro_rms, acc_hp_rms):
    def _u():
        stable_var.set(str(stable))
        current_var.set(str(current))
        conf_var.set(f"{conf:.2f}")
        rows_var.set(str(nrows))
        gyro_var.set(f"{gyro_rms:.1f}")
        acchp_var.set(f"{acc_hp_rms:.1f}")
    root.after(0, _u)

def start_ui_state():
    def _u():
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")
    root.after(0, _u)

def stop_ui_state():
    def _u():
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")
        mouse_active_btn.config(state="disabled") 
    root.after(0, _u)

def get_selected_address():
    selected = device_choice_var.get().strip()
    if selected in device_map: return device_map[selected]
    manual = manual_addr_var.get().strip()
    if manual: return manual
    return DEFAULT_BLE_ADDRESS

def toggle_mouse(event=None):
    global mouse_active
    if not collecting: return "break"
    mouse_active = not mouse_active
    if mouse_active:
        mouse_active_btn.config(text="Mouse Control: ON", style="On.TButton")
        set_status("Mouse Control Enabled.")
    else:
        mouse_active_btn.config(text="Mouse Control: OFF", style="Off.TButton")
        set_status("Mouse Control Disabled.")
    return "break"

def scan_devices():
    global scanning
    if scanning: return
    scanning = True
    set_status("Scanning BLE devices...")
    def _scan():
        global device_map, scanning
        try:
            devices = asyncio.run(BleakScanner.discover(timeout=6.0))
            temp, opts = {}, []
            for d in devices:
                name = (d.name or "Unknown").strip()
                addr = d.address
                label = f"{name} ({addr})"
                temp[label] = addr
                opts.append(label)
            opts.sort(key=lambda x: x.lower())
            device_map = temp
            if opts:
                update_device_dropdown(opts)
                set_status(f"Found {len(opts)} devices. Select one and press F5.")
            else: set_status("No BLE devices found.")
        except Exception as e: set_status(f"Scan failed: {e}")
        finally: scanning = False
    threading.Thread(target=_scan, daemon=True).start()

def start_stream(event=None):
    global collecting, ble_thread, stable_label, last_pred_label, same_count
    global idle_time_ms, cursor_freeze_time, click_capture_active
    if collecting: return "break"
    address = get_selected_address()
    if not address:
        set_status("Select a device (Scan) or enter MAC manually.")
        return "break"
    with buf_lock: buf.clear()
    fast_sensor_buf.clear()
    stable_label, last_pred_label, same_count = "—", None, 0
    idle_time_ms = 0
    cursor_freeze_time = 0.0
    click_capture_active = False
    collecting = True
    start_ui_state()
    mouse_active_btn.config(state="normal") 
    set_status(f"Connecting to {address}...")
    ble_thread = threading.Thread(target=run_ble_loop, args=(address,), daemon=True)
    ble_thread.start()
    return "break"

def stop_stream(event=None):
    global collecting, mouse_active
    collecting = False
    if mouse_active: toggle_mouse() 
    set_status("Stopping...")
    return "break"

# =========================
# BUILD UI
# =========================
root = tk.Tk()
root.title("Real-time Gesture Predictor & ML Mouse Control")

style = ttk.Style()
style.configure("On.TButton", foreground="green", font=("Segoe UI", 10, "bold"))
style.configure("Off.TButton", foreground="red", font=("Segoe UI", 10, "bold"))

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="nsew")
root.resizable(False, False)

ttk.Label(frame, text="Controls: F5 Start | F6 Stop | F7 Toggle Mouse").grid(row=0, column=0, columnspan=3, sticky="w")
ttk.Label(frame, text="Gesture Model:").grid(row=1, column=0, sticky="w", pady=(6, 0))
ttk.Label(frame, text=GESTURE_MODEL_PATH).grid(row=1, column=1, sticky="w", pady=(6, 0))
ttk.Label(frame, text="Click Model:").grid(row=2, column=0, sticky="w", pady=(2, 0))
ttk.Label(frame, text=CLICK_MODEL_PATH).grid(row=2, column=1, sticky="w", pady=(2, 0))

ttk.Label(frame, text="BLE Device:").grid(row=3, column=0, sticky="w", pady=(8, 2))
device_choice_var = tk.StringVar(value="")
device_combo = ttk.Combobox(frame, width=38, textvariable=device_choice_var, state="readonly")
device_combo.grid(row=3, column=1, sticky="w", pady=(8, 2))
ttk.Button(frame, text="Scan", command=scan_devices).grid(row=3, column=2, padx=(8, 0), pady=(8, 2))

ttk.Label(frame, text="Manual MAC:").grid(row=4, column=0, sticky="w")
manual_addr_var = tk.StringVar(value="")
ttk.Entry(frame, width=22, textvariable=manual_addr_var).grid(row=4, column=1, sticky="w")

start_btn = ttk.Button(frame, text="Start (F5)", command=start_stream)
start_btn.grid(row=5, column=0, pady=10)
stop_btn = ttk.Button(frame, text="Stop (F6)", command=stop_stream, state="disabled")
stop_btn.grid(row=5, column=1, pady=10, sticky="w")
mouse_active_btn = ttk.Button(frame, text="Mouse Control: OFF", command=toggle_mouse, state="disabled", style="Off.TButton")
mouse_active_btn.grid(row=5, column=2, padx=(8, 0), pady=10)

panel = ttk.LabelFrame(frame, text="Prediction & Stats", padding="10")
panel.grid(row=6, column=0, columnspan=3, sticky="we", pady=(6, 6))

stable_var = tk.StringVar(value="—")
current_var = tk.StringVar(value="—")
conf_var = tk.StringVar(value="0.00")
rows_var = tk.StringVar(value="0")
gyro_var = tk.StringVar(value="0.0")
acchp_var = tk.StringVar(value="0.0")
mouse_speed_var = tk.StringVar(value="0.0")
jerk_var = tk.StringVar(value="0.0")

ttk.Label(panel, text="Stable label:").grid(row=0, column=0, sticky="w")
ttk.Label(panel, textvariable=stable_var, font=("Segoe UI", 14, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))
ttk.Label(panel, text="Current label:").grid(row=1, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=current_var).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Confidence:").grid(row=2, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=conf_var).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Buffer rows:").grid(row=3, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=rows_var).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Gyro RMS:").grid(row=4, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=gyro_var).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Acc HP RMS:").grid(row=5, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=acchp_var).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Mouse Speed:").grid(row=6, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=mouse_speed_var).grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
ttk.Label(panel, text="Last Accel Jerk:").grid(row=7, column=0, sticky="w", pady=(6, 0))
ttk.Label(panel, textvariable=jerk_var, foreground="blue").grid(row=7, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

status_var = tk.StringVar(value="Ready. Scan/select device then press F5.")
ttk.Label(frame, textvariable=status_var, wraplength=560).grid(row=8, column=0, columnspan=3, sticky="w", pady=(10,0))

root.bind_all("<F5>", start_stream)
root.bind_all("<F6>", stop_stream)
root.bind_all("<F7>", toggle_mouse)  

def on_close():
    global collecting; collecting = False; root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(PREDICT_EVERY_MS, prediction_tick)
root.after(MOUSE_UPDATE_MS, mouse_tick)  
root.mainloop()