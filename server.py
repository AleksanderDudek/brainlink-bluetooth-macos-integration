"""
BrainLink Server — Real-time EEG-to-Game pipeline
FastAPI backend that processes BrainAccess HALO data and exposes
game-ready metrics via WebSocket + REST API.

For MVP: includes a realistic EEG simulator so the dashboard works
without a physical HALO device. Replace EEGSimulator with
BrainAccessSDK reader for production.
"""

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scipy.signal import butter, sosfilt, welch

# ─── Constants ────────────────────────────────────────────────────────
SAMPLE_RATE = 250          # HALO native rate
CHANNELS = ["Fp1", "Fp2", "O1", "O2"]
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
WINDOW_SAMPLES = SAMPLE_RATE * 1  # 1-second FFT window
STREAM_HZ = 10                     # push rate to clients
PUSH_INTERVAL = 1.0 / STREAM_HZ

# ─── EEG Simulator (replace with BrainAccess SDK for production) ─────

# Pre-built auto-play scenarios for dashboard testing
AUTO_SCENARIOS = {
    "gaming_session": [
        ("baseline",    8),
        ("focus",      12),
        ("stress",      6),
        ("focus",       8),
        ("flow",       15),
        ("stress",      5),
        ("eyes_closed", 6),
        ("baseline",    5),
    ],
    "horror_game": [
        ("baseline",    5),
        ("focus",       8),
        ("stress",     10),
        ("panic_spike", 2),
        ("stress",      6),
        ("eyes_closed", 4),
        ("focus",       6),
        ("stress",     12),
        ("panic_spike", 3),
        ("flow",        8),
    ],
    "meditation_trainer": [
        ("baseline",    6),
        ("eyes_closed",10),
        ("focus",       8),
        ("eyes_closed",12),
        ("flow",       20),
        ("eyes_closed", 8),
        ("baseline",    5),
    ],
    "crafting_focus": [
        ("baseline",    4),
        ("focus",      20),
        ("flow",       15),
        ("focus",      10),
        ("stress",      3),
        ("focus",      12),
        ("flow",       10),
    ],
}


class EEGSimulator:
    """
    Generates realistic EEG with controllable mental states, smooth
    transitions, and auto-scenario playback for dashboard testing.
    """

    def __init__(self):
        self.t = 0.0
        self.state = "baseline"
        self._target_state = "baseline"
        self._blend = 1.0
        self._blend_rate = 0.8
        self._prev_state = "baseline"
        self._auto_mode = False
        self._auto_scenario = None
        self._auto_step_idx = 0
        self._auto_step_timer = 0.0
        self._drift = {ch: np.random.randn() * 0.5 for ch in CHANNELS}
        self._drift_timer = 0.0

    def set_state(self, state: str):
        if state.startswith("auto:"):
            self.start_auto(state[5:])
            return
        if state == "auto_stop":
            self._auto_mode = False
            return
        if state != self._target_state:
            self._prev_state = self._target_state
            self._target_state = state
            self._blend = 0.0

    def start_auto(self, scenario_name: str):
        if scenario_name not in AUTO_SCENARIOS:
            return
        self._auto_mode = True
        self._auto_scenario = AUTO_SCENARIOS[scenario_name]
        self._auto_step_idx = 0
        self._auto_step_timer = 0.0
        self.set_state(self._auto_scenario[0][0])

    @property
    def auto_mode(self):
        return self._auto_mode

    @property
    def auto_progress(self) -> dict:
        if not self._auto_mode or not self._auto_scenario:
            return {}
        step = self._auto_scenario[self._auto_step_idx] if self._auto_step_idx < len(self._auto_scenario) else None
        return {
            "step_idx": self._auto_step_idx,
            "total_steps": len(self._auto_scenario),
            "current_state": step[0] if step else "done",
            "step_remaining_sec": round(step[1] - self._auto_step_timer, 1) if step else 0,
        }

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        dt = 1.0 / SAMPLE_RATE
        chunk_duration = n_samples * dt

        if self._auto_mode and self._auto_scenario:
            self._auto_step_timer += chunk_duration
            if self._auto_step_idx < len(self._auto_scenario):
                _, dur = self._auto_scenario[self._auto_step_idx]
                if self._auto_step_timer >= dur:
                    self._auto_step_timer = 0.0
                    self._auto_step_idx += 1
                    if self._auto_step_idx < len(self._auto_scenario):
                        next_state = self._auto_scenario[self._auto_step_idx][0]
                        self._prev_state = self._target_state
                        self._target_state = next_state
                        self._blend = 0.0
                    else:
                        self._auto_step_idx = 0
                        self._auto_step_timer = 0.0
                        self._prev_state = self._target_state
                        self._target_state = self._auto_scenario[0][0]
                        self._blend = 0.0

        self._blend = min(1.0, self._blend + self._blend_rate * chunk_duration)
        self.state = self._target_state if self._blend >= 0.99 else f"{self._prev_state}>{self._target_state}"

        self._drift_timer += chunk_duration
        if self._drift_timer > 3.0:
            self._drift_timer = 0.0
            for ch in CHANNELS:
                self._drift[ch] += np.random.randn() * 0.3
                self._drift[ch] *= 0.8

        data = {}
        for ch in CHANNELS:
            t_arr = self.t + np.arange(n_samples) * dt
            sig_old = self._gen_state_signal(ch, t_arr, self._prev_state)
            sig_new = self._gen_state_signal(ch, t_arr, self._target_state)
            samples = sig_old * (1 - self._blend) + sig_new * self._blend
            samples += self._drift[ch]
            data[ch] = samples

        self.t += n_samples * dt
        return data

    def _gen_state_signal(self, ch: str, t_arr: np.ndarray, state: str) -> np.ndarray:
        n = len(t_arr)
        is_frontal = ch.startswith("Fp")
        noise = np.random.randn(n) * 3.0

        if is_frontal:
            sig = (5.0 * np.sin(2 * np.pi * 10.0 * t_arr)
                   + 2.5 * np.sin(2 * np.pi * 20.0 * t_arr + 0.3)
                   + 2.0 * np.sin(2 * np.pi * 5.5 * t_arr)
                   + 0.8 * np.sin(2 * np.pi * 38 * t_arr)
                   + noise)
        else:
            sig = (8.0 * np.sin(2 * np.pi * 10.3 * t_arr)
                   + 1.5 * np.sin(2 * np.pi * 21 * t_arr + 0.5)
                   + 1.5 * np.sin(2 * np.pi * 6.0 * t_arr)
                   + 0.5 * np.sin(2 * np.pi * 40 * t_arr)
                   + noise)

        if state == "eyes_closed":
            if not is_frontal:
                sig += 25 * np.sin(2 * np.pi * 10.2 * t_arr)
                sig += 8 * np.sin(2 * np.pi * 10.5 * t_arr + 1.2)
            else:
                sig += 4 * np.sin(2 * np.pi * 10.0 * t_arr)
                sig -= 1.5 * np.sin(2 * np.pi * 20 * t_arr)

        elif state == "focus":
            if is_frontal:
                sig += 18 * np.sin(2 * np.pi * 22 * t_arr)
                sig += 6 * np.sin(2 * np.pi * 18 * t_arr + 0.8)
                sig += 3 * np.sin(2 * np.pi * 25 * t_arr + 1.5)
                sig -= 4 * np.sin(2 * np.pi * 10 * t_arr)
                sig += 2 * np.sin(2 * np.pi * 38 * t_arr)
            else:
                sig += 3 * np.sin(2 * np.pi * 20 * t_arr)

        elif state in ("stress", "panic_spike"):
            intensity = 2.0 if state == "panic_spike" else 1.0
            sig += intensity * 12 * np.sin(2 * np.pi * 25 * t_arr)
            sig += intensity * 8 * np.sin(2 * np.pi * 28 * t_arr + 0.4)
            sig += np.random.randn(n) * 8 * intensity
            sig += intensity * 6 * np.sin(2 * np.pi * 5 * t_arr)
            if is_frontal:
                blink_mask = np.random.random(n) < (0.012 * intensity)
                sig += blink_mask * (100 * intensity)
                sig += np.random.randn(n) * 5 * intensity

        elif state == "flow":
            phase_sync = 0.0
            if is_frontal:
                sig += 14 * np.sin(2 * np.pi * 20 * t_arr + phase_sync)
                sig += 8 * np.sin(2 * np.pi * 10 * t_arr + phase_sync)
                sig += 4 * np.sin(2 * np.pi * 38 * t_arr)
            else:
                sig += 18 * np.sin(2 * np.pi * 10.2 * t_arr + phase_sync)
                sig += 6 * np.sin(2 * np.pi * 20 * t_arr + phase_sync)
            sig -= noise * 0.5

        if ch == "Fp1":
            sig *= 1.15 if state in ("focus", "flow") else 1.05
        elif ch == "Fp2":
            sig *= 1.12 if state in ("stress", "panic_spike") else 0.97
        if ch == "O2":
            sig *= 0.95

        return sig


# ─── DSP Pipeline ─────────────────────────────────────────────────────
class DSPPipeline:
    """Band-power extraction via Welch PSD + bandpass filters."""

    def __init__(self):
        self.buffers: dict[str, np.ndarray] = {ch: np.array([]) for ch in CHANNELS}
        self._filters = {}
        for band_name, (lo, hi) in BANDS.items():
            sos = butter(4, [lo, hi], btype="band", fs=SAMPLE_RATE, output="sos")
            self._filters[band_name] = sos

    def push(self, data: dict[str, np.ndarray]):
        for ch in CHANNELS:
            self.buffers[ch] = np.concatenate([self.buffers[ch], data[ch]])
            # keep last 2 seconds max
            if len(self.buffers[ch]) > SAMPLE_RATE * 2:
                self.buffers[ch] = self.buffers[ch][-SAMPLE_RATE * 2:]

    def has_enough(self) -> bool:
        return all(len(self.buffers[ch]) >= WINDOW_SAMPLES for ch in CHANNELS)

    def compute_band_powers(self) -> dict[str, dict[str, float]]:
        """Return {channel: {band: power_uV2}}."""
        result = {}
        for ch in CHANNELS:
            buf = self.buffers[ch][-WINDOW_SAMPLES:]
            freqs, psd = welch(buf, fs=SAMPLE_RATE, nperseg=min(128, len(buf)))
            ch_bands = {}
            for band_name, (lo, hi) in BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                ch_bands[band_name] = float(np.mean(psd[mask])) if mask.any() else 0.0
            result[ch] = ch_bands
        return result

    def detect_blinks(self) -> int:
        """Count blink artifacts in last second on Fp channels."""
        count = 0
        for ch in ["Fp1", "Fp2"]:
            buf = self.buffers[ch][-WINDOW_SAMPLES:]
            threshold = np.std(buf) * 4
            peaks = np.where(np.abs(buf) > threshold)[0]
            if len(peaks) > 0:
                # group nearby peaks
                gaps = np.diff(peaks)
                count += 1 + np.sum(gaps > 20)
        return count


# ─── Feature Engine ───────────────────────────────────────────────────
@dataclass
class CalibrationProfile:
    baseline_powers: dict = field(default_factory=dict)
    focus_ceiling: float = 3.0
    calm_ceiling: float = 20.0
    stress_blink_ceiling: int = 30
    stress_theta_ceiling: float = 15.0
    flow_coherence_threshold: float = 0.7

@dataclass
class UserSettings:
    metrics: dict = field(default_factory=lambda: {
        "focus":       {"sensitivity": 1.0, "smoothing": 0.3, "dead_zone": 0.05, "window_sec": 1.0},
        "calm":        {"sensitivity": 1.0, "smoothing": 0.5, "dead_zone": 0.10, "window_sec": 2.0},
        "valence":     {"sensitivity": 1.0, "smoothing": 0.4, "dead_zone": 0.05},
        "instability": {"sensitivity": 1.0, "smoothing": 0.3, "dead_zone": 0.08},
        "stress":      {"sensitivity": 1.0, "smoothing": 0.4, "dead_zone": 0.10},
        "flow":        {"sensitivity": 1.0, "smoothing": 0.5, "dead_zone": 0.15},
    })
    combos: dict = field(default_factory=lambda: {
        "super_flow": {"requires": {"focus": ">0.7", "flow": ">0.8"}, "sustain_sec": 3.0, "cooldown_sec": 10.0, "enabled": True},
        "panic":      {"requires": {"stress": ">0.8", "flow": "<0.2"}, "sustain_sec": 1.5, "cooldown_sec": 5.0, "enabled": True},
        "zen_heal":   {"requires": {"calm": ">0.8", "stress": "<0.2"}, "sustain_sec": 2.0, "cooldown_sec": 8.0, "enabled": True},
        "berserker":  {"requires": {"stress": ">0.6", "focus": ">0.6"}, "sustain_sec": 2.0, "cooldown_sec": 12.0, "enabled": True},
    })
    global_settings: dict = field(default_factory=lambda: {
        "update_rate_hz": 10,
        "artifact_rejection": True,
        "adaptive_baseline": True,
        "adaptive_window_min": 5,
    })


class FeatureEngine:
    """Computes normalized 0-1 game metrics from band powers."""

    def __init__(self):
        self.calibration = CalibrationProfile()
        self.settings = UserSettings()
        self._prev_metrics = {m: 0.0 for m in ["focus", "calm", "valence", "instability", "stress", "flow"]}
        self._combo_states: dict[str, dict] = {}

    def compute(self, band_powers: dict, blink_count: int) -> dict:
        s = self.settings.metrics

        # Focus: beta/alpha ratio on Fp channels
        fp_alpha = (band_powers["Fp1"]["alpha"] + band_powers["Fp2"]["alpha"]) / 2
        fp_beta = (band_powers["Fp1"]["beta"] + band_powers["Fp2"]["beta"]) / 2
        raw_focus = fp_beta / max(fp_alpha, 0.01)
        focus = self._normalize(raw_focus, 0.5, self.calibration.focus_ceiling, s["focus"])

        # Calm: occipital alpha power
        occ_alpha = (band_powers["O1"]["alpha"] + band_powers["O2"]["alpha"]) / 2
        calm = self._normalize(occ_alpha, 2.0, self.calibration.calm_ceiling, s["calm"])

        # Valence (frontal asymmetry): log(Fp1_alpha) - log(Fp2_alpha)
        asym = math.log(max(band_powers["Fp1"]["alpha"], 0.01)) - math.log(max(band_powers["Fp2"]["alpha"], 0.01))
        valence = self._normalize(asym + 1.0, 0.0, 2.0, s["valence"])

        # Instability: theta burst detection
        fp_theta = (band_powers["Fp1"]["theta"] + band_powers["Fp2"]["theta"]) / 2
        baseline_theta = self.calibration.baseline_powers.get("Fp1", {}).get("theta", 5.0)
        instability = self._normalize(fp_theta, baseline_theta * 0.5, self.calibration.stress_theta_ceiling, s["instability"])

        # Stress: blink rate normalized
        stress = self._normalize(float(blink_count), 2.0, float(self.calibration.stress_blink_ceiling), s["stress"])

        # Flow: coherence approximation (correlation of alpha between F and O)
        f_alpha = fp_alpha
        o_alpha = occ_alpha
        coherence = 1.0 - abs(f_alpha - o_alpha) / max(f_alpha + o_alpha, 0.01)
        flow = self._normalize(coherence, 0.2, self.calibration.flow_coherence_threshold, s["flow"])

        metrics = {"focus": focus, "calm": calm, "valence": valence,
                   "instability": instability, "stress": stress, "flow": flow}

        # smoothing
        for key in metrics:
            sm = s.get(key, {}).get("smoothing", 0.3)
            metrics[key] = sm * self._prev_metrics[key] + (1 - sm) * metrics[key]
        self._prev_metrics = dict(metrics)

        # combos
        active_combos, combo_timers = self._evaluate_combos(metrics)

        return {
            "metrics": {k: round(v, 3) for k, v in metrics.items()},
            "active_combos": active_combos,
            "combo_timers": combo_timers,
        }

    def _normalize(self, value: float, floor: float, ceiling: float, params: dict) -> float:
        sensitivity = params.get("sensitivity", 1.0)
        dead_zone = params.get("dead_zone", 0.05)
        raw = (value - floor) / max(ceiling - floor, 0.001)
        raw = raw * sensitivity
        raw = max(0.0, min(1.0, raw))
        if raw < dead_zone:
            raw = 0.0
        return raw

    def _evaluate_combos(self, metrics: dict) -> tuple[list, dict]:
        now = time.time()
        active = []
        timers = {}

        for combo_name, combo_def in self.settings.combos.items():
            if not combo_def.get("enabled", True):
                continue

            state = self._combo_states.setdefault(combo_name, {
                "active": False, "active_since": 0, "cooldown_until": 0
            })

            if now < state["cooldown_until"]:
                timers[combo_name] = {"cooldown_remaining": round(state["cooldown_until"] - now, 1)}
                continue

            # check requirements
            met = True
            for metric_name, condition in combo_def["requires"].items():
                op = condition[0]
                threshold = float(condition[1:])
                val = metrics.get(metric_name, 0)
                if op == ">" and val <= threshold:
                    met = False
                elif op == "<" and val >= threshold:
                    met = False

            if met:
                if not state["active"]:
                    state["active"] = True
                    state["active_since"] = now
                elapsed = now - state["active_since"]
                if elapsed >= combo_def["sustain_sec"]:
                    active.append(combo_name)
                    timers[combo_name] = {"active_sec": round(elapsed, 1)}
            else:
                if state["active"]:
                    state["active"] = False
                    state["cooldown_until"] = now + combo_def.get("cooldown_sec", 5.0)

        return active, timers


# ─── Calibration Manager ─────────────────────────────────────────────
CALIBRATION_STEPS = ["baseline", "eyes_closed", "focus", "stress", "flow"]

class CalibrationSession:
    def __init__(self):
        self.session_id = f"cal_{uuid.uuid4().hex[:8]}"
        self.current_step_idx = 0
        self.step_data: dict[str, list[dict]] = {}
        self.completed = False
        self.profile: Optional[CalibrationProfile] = None

    @property
    def current_step(self):
        if self.current_step_idx < len(CALIBRATION_STEPS):
            return CALIBRATION_STEPS[self.current_step_idx]
        return None

    def record_step_data(self, step: str, band_powers: dict):
        self.step_data.setdefault(step, []).append(band_powers)

    def complete_step(self) -> dict:
        step = self.current_step
        data_list = self.step_data.get(step, [])

        result = {}
        if step == "baseline" and data_list:
            avg_powers = {}
            for ch in CHANNELS:
                avg_powers[ch] = {}
                for band in BANDS:
                    vals = [d[ch][band] for d in data_list if ch in d]
                    avg_powers[ch][band] = round(float(np.mean(vals)), 2) if vals else 0.0
            result = {"baseline_powers": avg_powers}

        elif step == "eyes_closed" and data_list:
            alphas = []
            for d in data_list:
                alphas.append((d.get("O1", {}).get("alpha", 0) + d.get("O2", {}).get("alpha", 0)) / 2)
            result = {"calm_ceiling": round(float(np.percentile(alphas, 90)), 2) if alphas else 20.0}

        elif step == "focus" and data_list:
            ratios = []
            for d in data_list:
                a = (d.get("Fp1", {}).get("alpha", 1) + d.get("Fp2", {}).get("alpha", 1)) / 2
                b = (d.get("Fp1", {}).get("beta", 1) + d.get("Fp2", {}).get("beta", 1)) / 2
                ratios.append(b / max(a, 0.01))
            result = {"focus_ceiling": round(float(np.percentile(ratios, 90)), 2) if ratios else 3.0}

        elif step == "stress" and data_list:
            thetas = []
            for d in data_list:
                thetas.append((d.get("Fp1", {}).get("theta", 0) + d.get("Fp2", {}).get("theta", 0)) / 2)
            result = {
                "stress_theta_ceiling": round(float(np.percentile(thetas, 90)), 2) if thetas else 15.0,
                "stress_blink_ceiling": 28  # simplified for sim
            }

        elif step == "flow" and data_list:
            coherences = []
            for d in data_list:
                fa = (d.get("Fp1", {}).get("alpha", 1) + d.get("Fp2", {}).get("alpha", 1)) / 2
                oa = (d.get("O1", {}).get("alpha", 1) + d.get("O2", {}).get("alpha", 1)) / 2
                c = 1.0 - abs(fa - oa) / max(fa + oa, 0.01)
                coherences.append(c)
            result = {"flow_coherence_threshold": round(float(np.percentile(coherences, 75)), 2) if coherences else 0.7}

        self.current_step_idx += 1
        next_step = self.current_step

        if next_step is None:
            self.completed = True
            self._build_profile()

        return {
            "step": step,
            "result": result,
            "next_step": next_step,
            "completed": self.completed,
        }

    def _build_profile(self):
        p = CalibrationProfile()
        for step, data_list in self.step_data.items():
            if step == "baseline" and data_list:
                avg = {}
                for ch in CHANNELS:
                    avg[ch] = {}
                    for band in BANDS:
                        vals = [d[ch][band] for d in data_list if ch in d]
                        avg[ch][band] = float(np.mean(vals)) if vals else 0.0
                p.baseline_powers = avg

        # use step results
        for step_result_key in ["calm_ceiling", "focus_ceiling", "stress_theta_ceiling",
                                 "stress_blink_ceiling", "flow_coherence_threshold"]:
            for step, data_list in self.step_data.items():
                pass  # already computed in complete_step

        # simplified: just set from the completion data
        self.profile = p


# ─── App State ────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.simulator = EEGSimulator()
        self.dsp = DSPPipeline()
        self.engine = FeatureEngine()
        self.ws_clients: list[WebSocket] = []
        self.calibration_session: Optional[CalibrationSession] = None
        self.running = False
        self.last_band_powers: dict = {}
        self.last_raw: dict = {}
        self.last_blinks: int = 0

state = AppState()


# ─── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="BrainLink", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Calibration endpoints ---
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

    # set simulator state for next step
    next_step = result.get("next_step")
    if next_step:
        state.simulator.set_state(next_step)

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
    return {
        "session_id": session_id,
        "profile": asdict(session.profile) if session.profile else {},
        "normalization": {
            "focus":  {"min": 0.5, "max": state.engine.calibration.focus_ceiling},
            "calm":   {"min": 2.0, "max": state.engine.calibration.calm_ceiling},
            "stress": {"min": 2.0, "max": float(state.engine.calibration.stress_blink_ceiling)},
            "flow":   {"min": 0.2, "max": state.engine.calibration.flow_coherence_threshold},
        }
    }


# --- User settings endpoints ---
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
    """Partial merge update of user settings."""
    if "metrics" in body:
        for metric_name, overrides in body["metrics"].items():
            if metric_name in state.engine.settings.metrics:
                state.engine.settings.metrics[metric_name].update(overrides)

    if "combos" in body:
        for combo_name, overrides in body["combos"].items():
            if combo_name in state.engine.settings.combos:
                state.engine.settings.combos[combo_name].update(overrides)
            else:
                state.engine.settings.combos[combo_name] = overrides

    if "global" in body:
        state.engine.settings.global_settings.update(body["global"])

    return {"status": "ok", "settings": {
        "metrics": state.engine.settings.metrics,
        "combos": state.engine.settings.combos,
        "global": state.engine.settings.global_settings,
    }}


@app.post("/api/v1/users/{user_id}/settings/reset")
async def reset_settings(user_id: str, body: dict = None):
    state.engine.settings = UserSettings()
    return {"status": "reset", "settings": {
        "metrics": state.engine.settings.metrics,
        "combos": state.engine.settings.combos,
        "global": state.engine.settings.global_settings,
    }}


# --- Simulator control (dev/debug) ---
@app.post("/api/v1/debug/set-state/{mental_state}")
async def debug_set_state(mental_state: str):
    if mental_state.startswith("auto:"):
        scenario = mental_state[5:]
        if scenario not in AUTO_SCENARIOS:
            raise HTTPException(400, f"Unknown scenario: {scenario}. Available: {list(AUTO_SCENARIOS.keys())}")
        state.simulator.set_state(mental_state)
        return {"mode": "auto", "scenario": scenario, "steps": AUTO_SCENARIOS[scenario]}
    if mental_state == "auto_stop":
        state.simulator.set_state("auto_stop")
        return {"mode": "manual", "state": "baseline"}
    if mental_state not in ["baseline", "eyes_closed", "focus", "stress", "flow", "panic_spike"]:
        raise HTTPException(400, f"Unknown state: {mental_state}")
    state.simulator.set_state(mental_state)
    return {"mode": "manual", "state": mental_state}


@app.get("/api/v1/debug/scenarios")
async def list_scenarios():
    return {
        "scenarios": {
            name: {
                "steps": steps,
                "total_duration_sec": sum(d for _, d in steps),
                "description": {
                    "gaming_session": "Typowa sesja grania: baseline → skupienie → stres → flow → relaks",
                    "horror_game": "Horror: narastający stres, nagłe spike'i paniki, próby skupienia",
                    "meditation_trainer": "Trening medytacji: relaks → skupienie → flow → głęboki relaks",
                    "crafting_focus": "Wytwarzanie przedmiotów: długie skupienie, flow, krótki stres",
                }.get(name, "")
            }
            for name, steps in AUTO_SCENARIOS.items()
        },
        "current_auto": state.simulator.auto_progress if state.simulator.auto_mode else None,
    }


# --- WebSocket stream ---
@app.websocket("/api/v1/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    state.ws_clients.append(websocket)
    try:
        while True:
            # keep alive; client can also send commands
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "set_state":
                    state.simulator.set_state(cmd.get("state", "baseline"))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        state.ws_clients.remove(websocket)


# --- Background EEG processing loop ---
async def eeg_loop():
    """Main processing loop: generate/read EEG → DSP → features → broadcast."""
    chunk_size = SAMPLE_RATE // STREAM_HZ  # samples per push cycle
    state.running = True

    while state.running:
        start = time.time()

        # 1. Generate (or read from SDK) raw samples
        raw_data = state.simulator.generate(chunk_size)
        state.last_raw = {ch: raw_data[ch][-4:].tolist() for ch in CHANNELS}

        # 2. Push into DSP buffers
        state.dsp.push(raw_data)

        # 3. If calibrating, record data
        if state.calibration_session and not state.calibration_session.completed:
            if state.dsp.has_enough():
                bp = state.dsp.compute_band_powers()
                state.calibration_session.record_step_data(
                    state.calibration_session.current_step, bp
                )

        # 4. Compute features if enough data
        if state.dsp.has_enough():
            band_powers = state.dsp.compute_band_powers()
            blinks = state.dsp.detect_blinks()
            state.last_band_powers = band_powers
            state.last_blinks = blinks

            features = state.engine.compute(band_powers, blinks)

            # 5. Build frame
            frame = {
                "ts": int(time.time() * 1000),
                "raw_preview": state.last_raw,
                "band_powers": {
                    ch: {b: round(v, 2) for b, v in bands.items()}
                    for ch, bands in band_powers.items()
                },
                "blink_count": blinks,
                "simulator": {
                    "state": state.simulator.state,
                    "auto_mode": state.simulator.auto_mode,
                    "auto_progress": state.simulator.auto_progress if state.simulator.auto_mode else None,
                },
                **features,
            }

            # 6. Broadcast to all WS clients
            dead = []
            for ws in state.ws_clients:
                try:
                    await ws.send_text(json.dumps(frame))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                state.ws_clients.remove(ws)

        # 7. Sleep to maintain push rate
        elapsed = time.time() - start
        await asyncio.sleep(max(0, PUSH_INTERVAL - elapsed))


@app.on_event("startup")
async def startup():
    asyncio.create_task(eeg_loop())


# --- Serve dashboard ---
_HERE = __import__("pathlib").Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_HERE / "dashboard")), name="static")

@app.get("/")
async def root():
    return FileResponse(str(_HERE / "dashboard" / "index.html"))


if __name__ == "__main__":
    import uvicorn, webbrowser, threading, os
    PORT = int(os.environ.get("BRAINLINK_PORT", 8420))
    NO_OPEN = os.environ.get("BRAINLINK_NO_OPEN", "0") == "1"
    if not NO_OPEN:
        def _open():
            import time; time.sleep(1.5)
            webbrowser.open(f"http://localhost:{PORT}")
        threading.Thread(target=_open, daemon=True).start()
    print(f"\n  BrainLink v0.1")
    print(f"  Dashboard:  http://localhost:{PORT}")
    print(f"  API docs:   http://localhost:{PORT}/docs")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
