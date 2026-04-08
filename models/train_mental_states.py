"""
Training pipeline for the BrainLink mental state classifier.

Produces a lightweight scikit-learn model that maps 4-channel band powers
to mental state labels, usable at 10 Hz in the real-time game loop.

Usage:
    python models/train_mental_states.py                # Train on synthetic data
    python models/train_mental_states.py --recordings   # Include BrainLink recordings
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TRAINED_DIR = ROOT / "trained"
DATA_DIR = ROOT / "data"
RECORDINGS_DIR = ROOT.parent / "recordings"
MODEL_PATH = TRAINED_DIR / "mental_state_clf.joblib"

# ── Constants (must match brainlink/constants.py) ──────────────────
CHANNELS = ["Fp1", "Fp2", "O1", "O2"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
SAMPLE_RATE = 250
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

MENTAL_STATES = [
    "neutral",
    "meditation",
    "focused",
    "relaxed",
    "visualization",
    "breathing",
    "drowsy",
    "stressed",
]


def extract_features(band_powers: dict) -> np.ndarray:
    """
    Extract the 25-feature vector from a band_powers dict.

    band_powers: {channel: {band: power}} for the 4 HALO channels.
    Returns: np.ndarray of shape (25,)
    """
    features = []

    # 20 raw band powers (4 channels × 5 bands)
    for ch in CHANNELS:
        for band in BANDS:
            features.append(band_powers.get(ch, {}).get(band, 0.0))

    # Frontal beta/alpha ratio
    fp_alpha = (band_powers["Fp1"]["alpha"] + band_powers["Fp2"]["alpha"]) / 2
    fp_beta = (band_powers["Fp1"]["beta"] + band_powers["Fp2"]["beta"]) / 2
    features.append(fp_beta / max(fp_alpha, 0.01))

    # Frontal theta/alpha ratio
    fp_theta = (band_powers["Fp1"]["theta"] + band_powers["Fp2"]["theta"]) / 2
    features.append(fp_theta / max(fp_alpha, 0.01))

    # Frontal alpha asymmetry (log Fp1 - log Fp2)
    asym = np.log(max(band_powers["Fp1"]["alpha"], 0.01)) - np.log(
        max(band_powers["Fp2"]["alpha"], 0.01)
    )
    features.append(asym)

    # Fronto-occipital alpha coherence
    occ_alpha = (band_powers["O1"]["alpha"] + band_powers["O2"]["alpha"]) / 2
    coherence = 1.0 - abs(fp_alpha - occ_alpha) / max(fp_alpha + occ_alpha, 0.01)
    features.append(coherence)

    # Alpha variability proxy (std across channels)
    alphas = [band_powers[ch]["alpha"] for ch in CHANNELS]
    features.append(float(np.std(alphas)))

    return np.array(features, dtype=np.float32)


# ── Synthetic training data generator ──────────────────────────────
# Until real datasets are downloaded, we generate synthetic band-power
# profiles that match known EEG signatures for each mental state.


def _band_powers(delta, theta, alpha, beta, gamma, noise=0.15):
    """Generate band powers for all 4 channels with noise."""
    rng = np.random.default_rng()
    result = {}
    for ch in CHANNELS:
        is_frontal = ch.startswith("F")
        scale = 1.0 if is_frontal else 0.9
        result[ch] = {
            "delta": max(0.1, delta * scale + rng.normal(0, noise)),
            "theta": max(0.1, theta * scale + rng.normal(0, noise)),
            "alpha": max(0.1, alpha * scale + rng.normal(0, noise)),
            "beta": max(0.1, beta * scale + rng.normal(0, noise)),
            "gamma": max(0.1, gamma * scale + rng.normal(0, noise)),
        }
    return result


# Typical EEG signatures per state (delta, theta, alpha, beta, gamma)
STATE_PROFILES = {
    "neutral": (3.0, 3.0, 5.0, 3.0, 1.0),
    "meditation": (2.0, 8.0, 10.0, 2.0, 0.5),     # high theta + alpha
    "focused": (1.5, 2.0, 3.0, 8.0, 2.0),          # high beta, low alpha
    "relaxed": (2.5, 3.0, 12.0, 2.5, 0.5),         # dominant alpha
    "visualization": (2.0, 6.0, 2.5, 4.0, 1.5),    # alpha suppression + theta
    "breathing": (2.0, 4.0, 8.0, 3.0, 0.5),        # rhythmic alpha
    "drowsy": (4.0, 7.0, 4.0, 1.5, 0.3),           # high theta, low beta
    "stressed": (2.0, 2.0, 2.0, 9.0, 3.0),         # high beta+gamma, low alpha
}


def generate_synthetic_data(n_per_class: int = 500) -> tuple:
    """Generate labeled training data from known EEG profiles."""
    X, y = [], []
    for label, (d, t, a, b, g) in STATE_PROFILES.items():
        for _ in range(n_per_class):
            bp = _band_powers(d, t, a, b, g, noise=0.3)
            feats = extract_features(bp)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)


def load_brainlink_recordings() -> tuple:
    """Load labeled recordings from BrainLink's own recording system."""
    X, y = [], []
    if not RECORDINGS_DIR.exists():
        return np.array(X), np.array(y)

    for meta_path in RECORDINGS_DIR.glob("*_meta.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        label = meta.get("mental_state_label")
        if not label or label not in MENTAL_STATES:
            continue
        csv_path = meta_path.with_name(meta_path.stem.replace("_meta", "_data") + ".csv")
        if not csv_path.exists():
            continue
        # This is a placeholder — would need full DSP pipeline to convert
        # raw samples into band powers. For now, skip raw recordings.
        print(f"  Found labeled recording: {meta.get('name')} → {label}")

    return np.array(X), np.array(y)


def train():
    """Train the mental state classifier and save it."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import joblib

    print("Generating synthetic training data...")
    X_syn, y_syn = generate_synthetic_data(n_per_class=800)
    print(f"  Synthetic samples: {len(X_syn)} ({len(MENTAL_STATES)} classes × 800)")

    X, y = X_syn, y_syn

    # Encode labels
    le = LabelEncoder()
    le.fit(MENTAL_STATES)
    y_enc = le.transform(y)

    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Final fit on all data
    clf.fit(X, y_enc)

    # Feature importances
    feat_names = []
    for ch in CHANNELS:
        for band in BANDS:
            feat_names.append(f"{ch}_{band}")
    feat_names += ["beta_alpha_ratio", "theta_alpha_ratio", "alpha_asymmetry",
                   "fo_coherence", "alpha_variability"]
    importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])
    print("\n  Top 10 features:")
    for name, imp in importances[:10]:
        print(f"    {name:25s} {imp:.4f}")

    # Save model + label encoder
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {"classifier": clf, "label_encoder": le, "feature_names": feat_names}
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features: {len(feat_names)}")
    print(f"  Size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BrainLink mental state classifier")
    parser.add_argument("--recordings", action="store_true", help="Include BrainLink recordings")
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install scikit-learn joblib")
        sys.exit(1)

    train()
