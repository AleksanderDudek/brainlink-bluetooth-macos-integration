"""EEG simulator with controllable mental states and auto-scenario playback."""

import numpy as np

from .constants import CHANNELS, SAMPLE_RATE

# ─── Auto-play scenario definitions ──────────────────────────────────
# Each entry is (mental_state, duration_seconds).
AUTO_SCENARIOS: dict[str, list[tuple[str, int]]] = {
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
        ("eyes_closed", 10),
        ("focus",       8),
        ("eyes_closed", 12),
        ("flow",        20),
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
    Generates realistic synthetic EEG with controllable mental states,
    smooth cross-state transitions, and auto-scenario playback.

    Provides the same ``generate(n_samples)`` interface as the hardware
    adapters so it can be swapped in transparently.
    """

    def __init__(self):
        self.t = 0.0
        self.state = "baseline"
        self._target_state = "baseline"
        self._blend = 1.0
        self._blend_rate = 0.8
        self._prev_state = "baseline"
        self._auto_mode = False
        self._auto_scenario: list[tuple[str, int]] | None = None
        self._auto_step_idx = 0
        self._auto_step_timer = 0.0
        self._drift = {ch: np.random.randn() * 0.5 for ch in CHANNELS}
        self._drift_timer = 0.0

    # ── State control ──────────────────────────────────────────────
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
    def auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def auto_progress(self) -> dict:
        if not self._auto_mode or not self._auto_scenario:
            return {}
        step = (
            self._auto_scenario[self._auto_step_idx]
            if self._auto_step_idx < len(self._auto_scenario)
            else None
        )
        return {
            "step_idx": self._auto_step_idx,
            "total_steps": len(self._auto_scenario),
            "current_state": step[0] if step else "done",
            "step_remaining_sec": round(step[1] - self._auto_step_timer, 1) if step else 0,
        }

    # ── Sample generation ──────────────────────────────────────────
    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        dt = 1.0 / SAMPLE_RATE
        chunk_duration = n_samples * dt

        self._advance_auto_scenario(chunk_duration)

        self._blend = min(1.0, self._blend + self._blend_rate * chunk_duration)
        self.state = (
            self._target_state
            if self._blend >= 0.99
            else f"{self._prev_state}>{self._target_state}"
        )

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
            data[ch] = sig_old * (1 - self._blend) + sig_new * self._blend + self._drift[ch]

        self.t += n_samples * dt
        return data

    def _advance_auto_scenario(self, chunk_duration: float):
        if not (self._auto_mode and self._auto_scenario):
            return
        self._auto_step_timer += chunk_duration
        if self._auto_step_idx >= len(self._auto_scenario):
            return
        _, dur = self._auto_scenario[self._auto_step_idx]
        if self._auto_step_timer < dur:
            return
        self._auto_step_timer = 0.0
        self._auto_step_idx += 1
        if self._auto_step_idx < len(self._auto_scenario):
            next_state = self._auto_scenario[self._auto_step_idx][0]
        else:
            # Loop back to start
            self._auto_step_idx = 0
            next_state = self._auto_scenario[0][0]
        self._prev_state = self._target_state
        self._target_state = next_state
        self._blend = 0.0

    def _gen_state_signal(self, ch: str, t_arr: np.ndarray, state: str) -> np.ndarray:
        n = len(t_arr)
        is_frontal = ch.startswith("Fp")
        noise = np.random.randn(n) * 3.0

        # Baseline signal (mixed alpha + beta + theta)
        if is_frontal:
            sig = (
                5.0 * np.sin(2 * np.pi * 10.0 * t_arr)
                + 2.5 * np.sin(2 * np.pi * 20.0 * t_arr + 0.3)
                + 2.0 * np.sin(2 * np.pi * 5.5 * t_arr)
                + 0.8 * np.sin(2 * np.pi * 38 * t_arr)
                + noise
            )
        else:
            sig = (
                8.0 * np.sin(2 * np.pi * 10.3 * t_arr)
                + 1.5 * np.sin(2 * np.pi * 21 * t_arr + 0.5)
                + 1.5 * np.sin(2 * np.pi * 6.0 * t_arr)
                + 0.5 * np.sin(2 * np.pi * 40 * t_arr)
                + noise
            )

        # State-specific modulations
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

        # Per-channel scaling for inter-electrode variability
        if ch == "Fp1":
            sig *= 1.15 if state in ("focus", "flow") else 1.05
        elif ch == "Fp2":
            sig *= 1.12 if state in ("stress", "panic_spike") else 0.97
        elif ch == "O2":
            sig *= 0.95

        return sig
