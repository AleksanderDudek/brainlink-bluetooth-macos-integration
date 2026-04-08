"""Runtime mental state classifier for the BrainLink real-time pipeline."""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("brainlink.mental_state")

CHANNELS = ["Fp1", "Fp2", "O1", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained" / "mental_state_clf.joblib"

# Mental state display metadata
STATE_META = {
    "neutral":       {"label": "Neutral",       "icon": "—",  "color": "#9a9890"},
    "meditation":    {"label": "Meditation",     "icon": "🧘", "color": "#a070ff"},
    "focused":       {"label": "Focused",        "icon": "🎯", "color": "#00e5a0"},
    "relaxed":       {"label": "Relaxed",        "icon": "🌊", "color": "#3d9eff"},
    "visualization": {"label": "Visualization",  "icon": "👁", "color": "#ff5ca0"},
    "breathing":     {"label": "Breathing",      "icon": "🌬", "color": "#00d4aa"},
    "drowsy":        {"label": "Drowsy",         "icon": "😴", "color": "#ffb020"},
    "stressed":      {"label": "Stressed",       "icon": "⚡", "color": "#ff4d5a"},
}


def _extract_features(band_powers: dict) -> np.ndarray:
    """Extract 25-feature vector from band_powers dict."""
    features = []
    for ch in CHANNELS:
        for band in BANDS:
            features.append(band_powers.get(ch, {}).get(band, 0.0))

    fp_alpha = (band_powers["Fp1"]["alpha"] + band_powers["Fp2"]["alpha"]) / 2
    fp_beta = (band_powers["Fp1"]["beta"] + band_powers["Fp2"]["beta"]) / 2
    fp_theta = (band_powers["Fp1"]["theta"] + band_powers["Fp2"]["theta"]) / 2
    occ_alpha = (band_powers["O1"]["alpha"] + band_powers["O2"]["alpha"]) / 2

    features.append(fp_beta / max(fp_alpha, 0.01))
    features.append(fp_theta / max(fp_alpha, 0.01))
    asym = np.log(max(band_powers["Fp1"]["alpha"], 0.01)) - np.log(
        max(band_powers["Fp2"]["alpha"], 0.01)
    )
    features.append(asym)
    coherence = 1.0 - abs(fp_alpha - occ_alpha) / max(fp_alpha + occ_alpha, 0.01)
    features.append(coherence)
    alphas = [band_powers[ch]["alpha"] for ch in CHANNELS]
    features.append(float(np.std(alphas)))

    return np.array(features, dtype=np.float32).reshape(1, -1)


class MentalStateClassifier:
    """Lightweight wrapper around the trained sklearn model for real-time use."""

    def __init__(self):
        self._clf = None
        self._le = None
        self._loaded = False
        self._current_state: str = "neutral"
        self._state_since: float = 0.0
        self._confidence: float = 0.0
        self._history: list[str] = []
        self._history_max = 10  # sliding window for smoothing
        self._min_confidence = 0.35  # below this → neutral
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            log.warning("Mental state model not found at %s — using rule-based fallback", MODEL_PATH)
            return
        try:
            import joblib
            bundle = joblib.load(MODEL_PATH)
            self._clf = bundle["classifier"]
            self._le = bundle["label_encoder"]
            self._loaded = True
            log.info("Mental state model loaded: %d classes", len(self._le.classes_))
        except Exception as exc:
            log.warning("Failed to load mental state model: %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, band_powers: dict) -> dict:
        """
        Classify current mental state from band powers.

        Returns dict with:
          - state: str (mental state label)
          - confidence: float (0-1)
          - duration_sec: float (how long in this state)
          - meta: dict (display info)
          - all_probs: dict (probability for each class)
        """
        if self._loaded:
            return self._classify_ml(band_powers)
        return self._classify_rules(band_powers)

    def _classify_ml(self, band_powers: dict) -> dict:
        features = _extract_features(band_powers)
        proba = self._clf.predict_proba(features)[0]
        classes = self._le.classes_

        # Build probability dict
        all_probs = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}

        # Get top prediction
        top_idx = int(np.argmax(proba))
        top_class = classes[top_idx]
        top_conf = float(proba[top_idx])

        # Smooth with history
        self._history.append(top_class)
        if len(self._history) > self._history_max:
            self._history.pop(0)

        # Majority vote from recent history
        from collections import Counter
        votes = Counter(self._history)
        smoothed_state = votes.most_common(1)[0][0]

        # Apply confidence threshold
        if top_conf < self._min_confidence:
            smoothed_state = "neutral"

        # Track state duration
        now = time.time()
        if smoothed_state != self._current_state:
            self._current_state = smoothed_state
            self._state_since = now
        self._confidence = top_conf

        return {
            "state": self._current_state,
            "confidence": round(self._confidence, 3),
            "duration_sec": round(now - self._state_since, 1),
            "meta": STATE_META.get(self._current_state, STATE_META["neutral"]),
            "all_probs": all_probs,
        }

    def _classify_rules(self, band_powers: dict) -> dict:
        """Fallback rule-based classification when no ML model is loaded."""
        fp_alpha = (band_powers["Fp1"]["alpha"] + band_powers["Fp2"]["alpha"]) / 2
        fp_beta = (band_powers["Fp1"]["beta"] + band_powers["Fp2"]["beta"]) / 2
        fp_theta = (band_powers["Fp1"]["theta"] + band_powers["Fp2"]["theta"]) / 2
        occ_alpha = (band_powers["O1"]["alpha"] + band_powers["O2"]["alpha"]) / 2
        fp_gamma = (band_powers["Fp1"]["gamma"] + band_powers["Fp2"]["gamma"]) / 2

        beta_alpha = fp_beta / max(fp_alpha, 0.01)
        theta_alpha = fp_theta / max(fp_alpha, 0.01)

        state = "neutral"
        confidence = 0.5

        if fp_theta > 6.0 and occ_alpha > 8.0:
            state, confidence = "meditation", 0.7
        elif beta_alpha > 2.0 and fp_theta < 3.0:
            state, confidence = "focused", 0.7
        elif occ_alpha > 9.0 and fp_beta < 3.5:
            state, confidence = "relaxed", 0.7
        elif fp_theta > 5.0 and fp_alpha < 4.0:
            state, confidence = "visualization", 0.6
        elif theta_alpha > 1.5 and fp_beta < 2.5:
            state, confidence = "drowsy", 0.6
        elif fp_beta > 7.0 and fp_gamma > 2.0 and fp_alpha < 3.0:
            state, confidence = "stressed", 0.7
        elif occ_alpha > 6.0 and fp_alpha > 6.0:
            state, confidence = "breathing", 0.5

        now = time.time()
        if state != self._current_state:
            self._current_state = state
            self._state_since = now
        self._confidence = confidence

        return {
            "state": self._current_state,
            "confidence": round(confidence, 3),
            "duration_sec": round(now - self._state_since, 1),
            "meta": STATE_META.get(self._current_state, STATE_META["neutral"]),
            "all_probs": {self._current_state: confidence},
        }
