"""Guided calibration session that builds per-user normalization profiles."""

import uuid
from typing import Optional

import numpy as np

from .constants import BANDS, CHANNELS
from .features import CalibrationProfile

CALIBRATION_STEPS = ["baseline", "eyes_closed", "focus", "stress", "flow"]


class CalibrationSession:
    """
    Collects band-power snapshots across five guided mental-state steps and
    derives a :class:`CalibrationProfile` for the :class:`~.features.FeatureEngine`.
    """

    def __init__(self):
        self.session_id = f"cal_{uuid.uuid4().hex[:8]}"
        self.current_step_idx = 0
        self.step_data: dict[str, list[dict]] = {}
        self.completed = False
        self.profile: Optional[CalibrationProfile] = None

    @property
    def current_step(self) -> Optional[str]:
        if self.current_step_idx < len(CALIBRATION_STEPS):
            return CALIBRATION_STEPS[self.current_step_idx]
        return None

    def record_step_data(self, step: str, band_powers: dict):
        """Append one band-power snapshot for the given step."""
        self.step_data.setdefault(step, []).append(band_powers)

    def complete_step(self) -> dict:
        """Advance to the next step and return a summary of what was measured."""
        step = self.current_step
        data_list = self.step_data.get(step, [])
        result = self._derive_step_result(step, data_list)

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

    # ── Internal ───────────────────────────────────────────────────
    def _derive_step_result(self, step: str, data_list: list) -> dict:
        if not data_list:
            return {}

        if step == "baseline":
            avg = {}
            for ch in CHANNELS:
                avg[ch] = {
                    band: round(float(np.mean([d[ch][band] for d in data_list if ch in d])), 2)
                    for band in BANDS
                }
            return {"baseline_powers": avg}

        if step == "eyes_closed":
            alphas = [
                (d.get("O1", {}).get("alpha", 0) + d.get("O2", {}).get("alpha", 0)) / 2
                for d in data_list
            ]
            return {"calm_ceiling": round(float(np.percentile(alphas, 90)), 2) if alphas else 20.0}

        if step == "focus":
            ratios = []
            for d in data_list:
                a = (d.get("Fp1", {}).get("alpha", 1) + d.get("Fp2", {}).get("alpha", 1)) / 2
                b = (d.get("Fp1", {}).get("beta",  1) + d.get("Fp2", {}).get("beta",  1)) / 2
                ratios.append(b / max(a, 0.01))
            return {"focus_ceiling": round(float(np.percentile(ratios, 90)), 2) if ratios else 3.0}

        if step == "stress":
            thetas = [
                (d.get("Fp1", {}).get("theta", 0) + d.get("Fp2", {}).get("theta", 0)) / 2
                for d in data_list
            ]
            return {
                "stress_theta_ceiling": round(float(np.percentile(thetas, 90)), 2) if thetas else 15.0,
                "stress_blink_ceiling": 28,
            }

        if step == "flow":
            coherences = []
            for d in data_list:
                fa = (d.get("Fp1", {}).get("alpha", 1) + d.get("Fp2", {}).get("alpha", 1)) / 2
                oa = (d.get("O1",  {}).get("alpha", 1) + d.get("O2",  {}).get("alpha", 1)) / 2
                coherences.append(1.0 - abs(fa - oa) / max(fa + oa, 0.01))
            return {"flow_coherence_threshold": round(float(np.percentile(coherences, 75)), 2) if coherences else 0.7}

        return {}

    def _build_profile(self):
        p = CalibrationProfile()
        baseline_data = self.step_data.get("baseline", [])
        if baseline_data:
            p.baseline_powers = {
                ch: {
                    band: float(np.mean([d[ch][band] for d in baseline_data if ch in d]))
                    for band in BANDS
                }
                for ch in CHANNELS
            }
        self.profile = p
