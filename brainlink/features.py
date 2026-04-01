"""Feature engine: band-power → normalized game metrics + combo detection."""

import math
import time
from dataclasses import dataclass, field


@dataclass
class CalibrationProfile:
    """Per-user normalization ceilings derived from a calibration session."""

    baseline_powers: dict = field(default_factory=dict)
    focus_ceiling: float = 3.0
    calm_ceiling: float = 20.0
    stress_blink_ceiling: int = 30
    stress_theta_ceiling: float = 15.0
    flow_coherence_threshold: float = 0.7


@dataclass
class UserSettings:
    """Per-metric sensitivity/smoothing knobs and combo definitions."""

    metrics: dict = field(default_factory=lambda: {
        "focus":       {"sensitivity": 1.0, "smoothing": 0.3, "dead_zone": 0.05, "window_sec": 1.0},
        "calm":        {"sensitivity": 1.0, "smoothing": 0.5, "dead_zone": 0.10, "window_sec": 2.0},
        "valence":     {"sensitivity": 1.0, "smoothing": 0.4, "dead_zone": 0.05},
        "instability": {"sensitivity": 1.0, "smoothing": 0.3, "dead_zone": 0.08},
        "stress":      {"sensitivity": 1.0, "smoothing": 0.4, "dead_zone": 0.10},
        "flow":        {"sensitivity": 1.0, "smoothing": 0.5, "dead_zone": 0.15},
    })
    combos: dict = field(default_factory=lambda: {
        "super_flow": {"requires": {"focus": ">0.7", "flow": ">0.8"},  "sustain_sec": 3.0,  "cooldown_sec": 10.0, "enabled": True},
        "panic":      {"requires": {"stress": ">0.8", "flow": "<0.2"}, "sustain_sec": 1.5,  "cooldown_sec": 5.0,  "enabled": True},
        "zen_heal":   {"requires": {"calm": ">0.8", "stress": "<0.2"}, "sustain_sec": 2.0,  "cooldown_sec": 8.0,  "enabled": True},
        "berserker":  {"requires": {"stress": ">0.6", "focus": ">0.6"},"sustain_sec": 2.0,  "cooldown_sec": 12.0, "enabled": True},
    })
    global_settings: dict = field(default_factory=lambda: {
        "update_rate_hz": 10,
        "artifact_rejection": True,
        "adaptive_baseline": True,
        "adaptive_window_min": 5,
    })


class FeatureEngine:
    """
    Converts band-power estimates into normalized [0, 1] game metrics
    and detects active multi-metric combos.
    """

    def __init__(self):
        self.calibration = CalibrationProfile()
        self.settings = UserSettings()
        self._prev: dict[str, float] = {
            m: 0.0 for m in ("focus", "calm", "valence", "instability", "stress", "flow")
        }
        self._combo_states: dict[str, dict] = {}

    def compute(self, band_powers: dict, blink_count: int) -> dict:
        """Return metrics dict with keys: metrics, active_combos, combo_timers."""
        s = self.settings.metrics
        c = self.calibration

        fp_alpha = (band_powers["Fp1"]["alpha"] + band_powers["Fp2"]["alpha"]) / 2
        fp_beta  = (band_powers["Fp1"]["beta"]  + band_powers["Fp2"]["beta"])  / 2
        occ_alpha = (band_powers["O1"]["alpha"] + band_powers["O2"]["alpha"]) / 2
        fp_theta  = (band_powers["Fp1"]["theta"] + band_powers["Fp2"]["theta"]) / 2

        # Focus: frontal beta/alpha ratio
        focus = self._normalize(fp_beta / max(fp_alpha, 0.01), 0.5, c.focus_ceiling, s["focus"])

        # Calm: occipital alpha power
        calm = self._normalize(occ_alpha, 2.0, c.calm_ceiling, s["calm"])

        # Valence: frontal alpha asymmetry (log Fp1 − log Fp2)
        asym = math.log(max(band_powers["Fp1"]["alpha"], 0.01)) - math.log(max(band_powers["Fp2"]["alpha"], 0.01))
        valence = self._normalize(asym + 1.0, 0.0, 2.0, s["valence"])

        # Instability: frontal theta relative to baseline
        baseline_theta = c.baseline_powers.get("Fp1", {}).get("theta", 5.0)
        instability = self._normalize(fp_theta, baseline_theta * 0.5, c.stress_theta_ceiling, s["instability"])

        # Stress: blink rate proxy
        stress = self._normalize(float(blink_count), 2.0, float(c.stress_blink_ceiling), s["stress"])

        # Flow: fronto-occipital alpha coherence approximation
        coherence = 1.0 - abs(fp_alpha - occ_alpha) / max(fp_alpha + occ_alpha, 0.01)
        flow = self._normalize(coherence, 0.2, c.flow_coherence_threshold, s["flow"])

        metrics = {
            "focus": focus, "calm": calm, "valence": valence,
            "instability": instability, "stress": stress, "flow": flow,
        }

        # Exponential smoothing
        for key in metrics:
            sm = s.get(key, {}).get("smoothing", 0.3)
            metrics[key] = sm * self._prev[key] + (1 - sm) * metrics[key]
        self._prev = dict(metrics)

        active_combos, combo_timers = self._evaluate_combos(metrics)
        return {
            "metrics": {k: round(v, 3) for k, v in metrics.items()},
            "active_combos": active_combos,
            "combo_timers": combo_timers,
        }

    # ── Internal helpers ───────────────────────────────────────────
    def _normalize(self, value: float, floor: float, ceiling: float, params: dict) -> float:
        sensitivity = params.get("sensitivity", 1.0)
        dead_zone = params.get("dead_zone", 0.05)
        raw = (value - floor) / max(ceiling - floor, 0.001) * sensitivity
        raw = max(0.0, min(1.0, raw))
        return 0.0 if raw < dead_zone else raw

    def _evaluate_combos(self, metrics: dict) -> tuple[list, dict]:
        now = time.time()
        active: list[str] = []
        timers: dict = {}

        for name, combo in self.settings.combos.items():
            if not combo.get("enabled", True):
                continue

            state = self._combo_states.setdefault(
                name, {"active": False, "active_since": 0.0, "cooldown_until": 0.0}
            )

            if now < state["cooldown_until"]:
                timers[name] = {"cooldown_remaining": round(state["cooldown_until"] - now, 1)}
                continue

            met = all(
                (metrics.get(m, 0) > float(cond[1:])) if cond[0] == ">"
                else (metrics.get(m, 0) < float(cond[1:]))
                for m, cond in combo["requires"].items()
            )

            if met:
                if not state["active"]:
                    state["active"] = True
                    state["active_since"] = now
                elapsed = now - state["active_since"]
                if elapsed >= combo["sustain_sec"]:
                    active.append(name)
                    timers[name] = {"active_sec": round(elapsed, 1)}
            else:
                if state["active"]:
                    state["active"] = False
                    state["cooldown_until"] = now + combo.get("cooldown_sec", 5.0)

        return active, timers
