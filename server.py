"""
BrainLink Server — Real-time EEG-to-Game pipeline.

FastAPI backend that reads from either the built-in EEG simulator or a
real BrainAccess HALO device and exposes game-ready metrics via:
  - WebSocket  /api/v1/stream   (10 Hz push)
  - REST API   /api/v1/…       (Swagger at /docs)
  - Dashboard  /               (static HTML)
"""

import asyncio
import csv
import json
import logging
import os
import time
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from brainlink.calibration import CalibrationSession, CALIBRATION_STEPS
from brainlink.constants import CHANNELS, PUSH_INTERVAL, RECORDINGS_DIR, SAMPLE_RATE, STREAM_HZ
from brainlink.dsp import DSPPipeline
from brainlink.features import FeatureEngine, UserSettings
from brainlink.recorder import SessionRecorder
from brainlink.simulator import EEGSimulator, AUTO_SCENARIOS

# ─── Optional hardware backends ──────────────────────────────────────
try:
    from brainlink.halo_sdk import BrainAccessHALO, HAS_BRAINACCESS
except ImportError:
    HAS_BRAINACCESS = False
    BrainAccessHALO = None  # type: ignore[assignment, misc]

try:
    from brainlink.halo_ble import HALOBleAdapter, HAS_BLEAK
    HAS_BLE_ADAPTER = HAS_BLEAK
except ImportError:
    HAS_BLE_ADAPTER = False
    HALOBleAdapter = None  # type: ignore[assignment, misc]

log = logging.getLogger("brainlink")


# ─── Application state ────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.simulator = EEGSimulator()
        # Prefer official SDK (Windows/Linux); fall back to BLE adapter (macOS)
        if HAS_BRAINACCESS:
            self.halo: Optional[BrainAccessHALO] = BrainAccessHALO()
            self.ble: Optional[HALOBleAdapter] = None
        elif HAS_BLE_ADAPTER:
            self.halo = None
            self.ble: Optional[HALOBleAdapter] = HALOBleAdapter()
        else:
            self.halo = None
            self.ble = None
        self.source = "simulator"   # "simulator" | "device" | "off"
        self.dsp = DSPPipeline()
        self.engine = FeatureEngine()
        self.ws_clients: list[WebSocket] = []
        self.calibration_session: Optional[CalibrationSession] = None
        self.running = False
        self.last_band_powers: dict = {}
        self.last_raw: dict = {}
        self.last_blinks: int = 0
        self.recorder = SessionRecorder()

    @property
    def device(self):
        """Return whichever device adapter is active (SDK preferred over BLE)."""
        return self.halo or self.ble

    @property
    def has_device_support(self) -> bool:
        return HAS_BRAINACCESS or HAS_BLE_ADAPTER


state = AppState()

# ─── FastAPI application ──────────────────────────────────────────────

app = FastAPI(title="BrainLink", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Calibration ──────────────────────────────────────────────────────

@app.post("/api/v1/calibration/start")
async def calibration_start():
    state.calibration_session = CalibrationSession()
    session = state.calibration_session
    state.simulator.set_state("baseline")
    return {
        "session_id": session.session_id,
        "steps": CALIBRATION_STEPS,
        "current_step": session.current_step,
        "estimated_duration_sec": 130,
    }


@app.post("/api/v1/calibration/{session_id}/step/{step_name}/complete")
async def calibration_complete_step(session_id: str, step_name: str):
    session = state.calibration_session
    if not session or session.session_id != session_id:
        raise HTTPException(404, "Calibration session not found")
    if session.current_step != step_name:
        raise HTTPException(400, f"Expected step '{session.current_step}', got '{step_name}'")
    result = session.complete_step()
    if result.get("next_step"):
        state.simulator.set_state(result["next_step"])
    if session.completed and session.profile:
        state.engine.calibration = session.profile
    return result


@app.get("/api/v1/calibration/{session_id}/profile")
async def calibration_profile(session_id: str):
    session = state.calibration_session
    if not session or session.session_id != session_id:
        raise HTTPException(404, "Calibration session not found")
    if not session.completed:
        raise HTTPException(400, "Calibration not completed yet")
    c = state.engine.calibration
    return {
        "session_id": session_id,
        "profile": asdict(session.profile) if session.profile else {},
        "normalization": {
            "focus":  {"min": 0.5, "max": c.focus_ceiling},
            "calm":   {"min": 2.0, "max": c.calm_ceiling},
            "stress": {"min": 2.0, "max": float(c.stress_blink_ceiling)},
            "flow":   {"min": 0.2, "max": c.flow_coherence_threshold},
        },
    }


# ─── User settings ────────────────────────────────────────────────────

@app.get("/api/v1/users/{user_id}/settings")
async def get_settings(user_id: str):
    return {
        "user_id": user_id,
        "metrics": state.engine.settings.metrics,
        "combos": state.engine.settings.combos,
        "global": state.engine.settings.global_settings,
    }


@app.patch("/api/v1/users/{user_id}/settings")
async def patch_settings(user_id: str, body: dict):
    """Partial merge — only keys present in body are updated."""
    s = state.engine.settings
    if "metrics" in body:
        for name, overrides in body["metrics"].items():
            if name in s.metrics:
                s.metrics[name].update(overrides)
    if "combos" in body:
        for name, overrides in body["combos"].items():
            if name in s.combos:
                s.combos[name].update(overrides)
            else:
                s.combos[name] = overrides
    if "global" in body:
        s.global_settings.update(body["global"])
    return {"status": "ok", "settings": {"metrics": s.metrics, "combos": s.combos, "global": s.global_settings}}


@app.post("/api/v1/users/{user_id}/settings/reset")
async def reset_settings(user_id: str, body: dict = None):
    state.engine.settings = UserSettings()
    s = state.engine.settings
    return {"status": "reset", "settings": {"metrics": s.metrics, "combos": s.combos, "global": s.global_settings}}


# ─── Device management ────────────────────────────────────────────────

@app.get("/api/v1/device/status")
async def device_status():
    dev = state.device
    result = {
        "sdk_available": HAS_BRAINACCESS,
        "ble_available": HAS_BLE_ADAPTER,
        "backend": "sdk" if HAS_BRAINACCESS else ("ble" if HAS_BLE_ADAPTER else "none"),
        "source": state.source,
        "simulator_state": state.simulator.state,
    }
    if dev and dev.is_connected:
        result["device"] = {**dev.device_info, "streaming": dev.is_streaming}
    return result


@app.post("/api/v1/device/scan")
async def device_scan():
    if not state.has_device_support:
        raise HTTPException(501, "No device backend available.")
    try:
        return {"devices": state.device.scan()}
    except Exception as exc:
        raise HTTPException(500, f"Scan failed: {exc}")


@app.post("/api/v1/device/connect")
async def device_connect(body: dict):
    if not state.has_device_support:
        raise HTTPException(501, "No device backend available.")
    name = body.get("device_name")
    if not name:
        raise HTTPException(400, "device_name is required")
    try:
        return state.device.connect(name)
    except Exception as exc:
        raise HTTPException(500, f"Connection failed: {exc}")


@app.post("/api/v1/device/disconnect")
async def device_disconnect():
    dev = state.device
    if dev and dev.is_connected:
        if state.source == "device":
            state.source = "off"
        dev.disconnect()
        return {"status": "disconnected", "source": state.source}
    return {"status": "not_connected"}


@app.post("/api/v1/device/start-stream")
async def device_start_stream():
    dev = state.device
    if not dev or not dev.is_connected:
        raise HTTPException(400, "No device connected")
    try:
        dev.start_stream()
        state.source = "device"
        return {"status": "streaming", "source": "device"}
    except Exception as exc:
        raise HTTPException(500, f"Failed to start stream: {exc}")


@app.post("/api/v1/device/stop-stream")
async def device_stop_stream():
    dev = state.device
    if dev and dev.is_streaming:
        dev.stop_stream()
    if state.source == "device":
        state.source = "off"
    return {"status": "stopped", "source": state.source}


@app.post("/api/v1/source/{source_name}")
async def set_source(source_name: str):
    if source_name not in ("simulator", "device", "off"):
        raise HTTPException(400, "source must be 'simulator', 'device', or 'off'")
    dev = state.device
    if source_name == "device":
        if not dev or not dev.is_connected:
            raise HTTPException(400, "No device connected — call /api/v1/device/connect first")
        if not dev.is_streaming:
            dev.start_stream()
    state.source = source_name
    return {"source": state.source}


# ─── Simulator / debug control ────────────────────────────────────────

_VALID_SIM_STATES = {"baseline", "eyes_closed", "focus", "stress", "flow", "panic_spike"}


@app.post("/api/v1/debug/set-state/{mental_state}")
async def debug_set_state(mental_state: str):
    if mental_state.startswith("auto:"):
        scenario = mental_state[5:]
        if scenario not in AUTO_SCENARIOS:
            raise HTTPException(400, f"Unknown scenario '{scenario}'. Available: {list(AUTO_SCENARIOS)}")
        state.simulator.set_state(mental_state)
        return {"mode": "auto", "scenario": scenario, "steps": AUTO_SCENARIOS[scenario]}
    if mental_state == "auto_stop":
        state.simulator.set_state("auto_stop")
        return {"mode": "manual", "state": "baseline"}
    if mental_state not in _VALID_SIM_STATES:
        raise HTTPException(400, f"Unknown state '{mental_state}'. Valid: {sorted(_VALID_SIM_STATES)}")
    state.simulator.set_state(mental_state)
    return {"mode": "manual", "state": mental_state}


@app.get("/api/v1/debug/scenarios")
async def list_scenarios():
    return {
        "scenarios": {
            name: {
                "steps": steps,
                "total_duration_sec": sum(d for _, d in steps),
            }
            for name, steps in AUTO_SCENARIOS.items()
        },
        "current_auto": state.simulator.auto_progress if state.simulator.auto_mode else None,
    }


# ─── Recording ────────────────────────────────────────────────────────

@app.post("/api/v1/recording/start")
async def recording_start(body: dict):
    name = body.get("name", "").strip() or f"session_{int(time.time())}"
    duration_sec = max(60, min(3600, int(body.get("duration_sec", 300))))
    if state.recorder.recording:
        raise HTTPException(400, "Recording already in progress")
    state.recorder.start(name, duration_sec, source=state.source)
    return state.recorder.info()


@app.post("/api/v1/recording/stop")
async def recording_stop():
    if not state.recorder.recording:
        raise HTTPException(400, "No recording in progress")
    return state.recorder.stop()


@app.patch("/api/v1/recording/notes")
async def recording_notes(body: dict):
    if not state.recorder.session_id:
        raise HTTPException(400, "No recording session")
    state.recorder.notes = body.get("notes", "")
    state.recorder._save_meta()
    return {"ok": True}


@app.get("/api/v1/recording/status")
async def recording_status():
    return state.recorder.info()


@app.get("/api/v1/recordings")
async def list_recordings():
    recs = []
    if RECORDINGS_DIR.exists():
        for meta_file in sorted(RECORDINGS_DIR.glob("*_meta.json"), reverse=True):
            with open(meta_file) as f:
                recs.append(json.load(f))
    return {"recordings": recs}


@app.get("/api/v1/recordings/{session_id}")
async def get_recording(session_id: str):
    meta_path = RECORDINGS_DIR / f"{session_id}_meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Recording not found")
    with open(meta_path) as f:
        meta = json.load(f)
    csv_path = RECORDINGS_DIR / f"{session_id}_data.csv"
    samples: dict[str, list] = {"ts": [], "Fp1": [], "Fp2": [], "O1": [], "O2": []}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                samples["ts"].append(float(row["timestamp"]))
                for ch in CHANNELS:
                    samples[ch].append(float(row[ch]))
    total = len(samples["ts"])
    if total > 2000:
        step = total // 2000
        samples = {k: v[::step] for k, v in samples.items()}
    meta["samples"] = samples
    return meta


@app.get("/api/v1/recordings/{session_id}/csv")
async def download_recording_csv(session_id: str):
    csv_path = RECORDINGS_DIR / f"{session_id}_data.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Recording CSV not found")
    meta_path = RECORDINGS_DIR / f"{session_id}_meta.json"
    filename = session_id
    if meta_path.exists():
        with open(meta_path) as f:
            filename = json.load(f).get("name", session_id).replace(" ", "_")
    return FileResponse(csv_path, media_type="text/csv", filename=f"{filename}.csv")


@app.delete("/api/v1/recordings/{session_id}")
async def delete_recording(session_id: str):
    meta_path = RECORDINGS_DIR / f"{session_id}_meta.json"
    csv_path = RECORDINGS_DIR / f"{session_id}_data.csv"
    if not meta_path.exists() and not csv_path.exists():
        raise HTTPException(404, "Recording not found")
    for p in (meta_path, csv_path):
        if p.exists():
            p.unlink()
    return {"ok": True, "deleted": session_id}


# ─── WebSocket stream ─────────────────────────────────────────────────

@app.websocket("/api/v1/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "set_state":
                    state.simulator.set_state(cmd.get("state", "baseline"))
            except json.JSONDecodeError:
                pass
    except (WebSocketDisconnect, Exception):
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


# ─── Background EEG processing loop ──────────────────────────────────

async def eeg_loop():
    """Generate/read EEG samples → DSP → features → broadcast at STREAM_HZ."""
    chunk_size = SAMPLE_RATE // STREAM_HZ
    state.running = True

    while state.running:
        t_start = time.time()
        try:
            await _process_cycle(state.device, chunk_size)
        except Exception:
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1.0)
        await asyncio.sleep(max(0.0, PUSH_INTERVAL - (time.time() - t_start)))


async def _process_cycle(dev, chunk_size: int):
    if state.source == "off":
        await _broadcast(json.dumps(_status_frame(dev), default=str))
        return

    if state.source == "device" and dev and dev.is_streaming:
        raw_data = dev.generate(chunk_size)
        if all(len(raw_data[ch]) == 0 for ch in CHANNELS):
            return  # device buffer not yet filled
    else:
        raw_data = state.simulator.generate(chunk_size)

    state.last_raw = {
        ch: raw_data[ch][-4:].tolist()
        for ch in CHANNELS
        if len(raw_data[ch]) > 0
    }

    if state.recorder.recording and state.recorder.add_samples(raw_data):
        log.info("Recording auto-stopped (duration reached)")

    state.dsp.push(raw_data)

    if (
        state.calibration_session
        and not state.calibration_session.completed
        and state.dsp.has_enough()
    ):
        bp = state.dsp.compute_band_powers()
        state.calibration_session.record_step_data(state.calibration_session.current_step, bp)

    if not state.dsp.has_enough():
        return

    band_powers = state.dsp.compute_band_powers()
    blinks = state.dsp.detect_blinks()
    state.last_band_powers = band_powers
    state.last_blinks = blinks

    features = state.engine.compute(band_powers, blinks)
    frame = {
        "ts": int(time.time() * 1000),
        "raw_preview": state.last_raw,
        "band_powers": {
            ch: {b: round(v, 2) for b, v in bands.items()}
            for ch, bands in band_powers.items()
        },
        "blink_count": blinks,
        "source": state.source,
        "simulator": (
            {
                "state": state.simulator.state,
                "auto_mode": state.simulator.auto_mode,
                "auto_progress": state.simulator.auto_progress if state.simulator.auto_mode else None,
            }
            if state.source == "simulator"
            else None
        ),
        "device": (
            {**dev.device_info, "connected": dev.is_connected, "streaming": dev.is_streaming}
            if dev
            else {"connected": False, "streaming": False}
        ),
        "recording": state.recorder.info() if state.recorder.recording else None,
        **features,
    }

    try:
        payload = json.dumps(frame, default=str, allow_nan=False)
    except (ValueError, TypeError):
        payload = json.dumps(frame, default=str)

    await _broadcast(payload)


def _status_frame(dev) -> dict:
    return {
        "ts": int(time.time() * 1000),
        "source": "off",
        "device": (
            {**dev.device_info, "connected": dev.is_connected, "streaming": dev.is_streaming}
            if dev
            else {"connected": False, "streaming": False}
        ),
        "recording": state.recorder.info() if state.recorder.recording else None,
    }


async def _broadcast(payload: str):
    dead = []
    for ws in state.ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)


# ─── Startup / static files ───────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(eeg_loop())


_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_HERE / "dashboard")), name="static")


@app.get("/")
async def root():
    return FileResponse(str(_HERE / "dashboard" / "index.html"))


# ─── Direct-run entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    PORT = int(os.environ.get("BRAINLINK_PORT", 8420))
    NO_OPEN = os.environ.get("BRAINLINK_NO_OPEN", "0") == "1"

    if not NO_OPEN:
        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{PORT}")
        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  BrainLink v0.1")
    print(f"  Dashboard:  http://localhost:{PORT}")
    print(f"  API docs:   http://localhost:{PORT}/docs")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
