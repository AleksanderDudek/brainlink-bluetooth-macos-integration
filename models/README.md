# BrainLink — Mental State Models

Trained models and training infrastructure for EEG-based mental state classification.

## Directory Structure

```
models/
├── README.md              ← This file
├── DATASETS.md            ← Curated data sources for training
├── download_datasets.py   ← Automated dataset downloader
├── train_mental_states.py ← Training pipeline
├── trained/               ← Serialized model weights (git-ignored)
│   └── .gitkeep
└── data/                  ← Raw downloaded datasets (git-ignored)
    └── .gitkeep
```

## Quick Start

```bash
# 1. Install training dependencies
pip install -r requirements-training.txt

# 2. Download datasets
python models/download_datasets.py --all

# 3. Train the mental state classifier
python models/train_mental_states.py

# 4. The trained model is saved to models/trained/mental_state_clf.joblib
#    and is automatically loaded by the BrainLink server on startup.
```

## Mental States Detected

| State | EEG Signature | Game Use |
|-------|--------------|----------|
| **Deep Meditation** | High theta (4-8 Hz), elevated frontal alpha | Shield / Heal |
| **Focused Attention** | High beta/alpha ratio, low theta | Power boost / Aim assist |
| **Relaxed / Calm** | Dominant occipital alpha (8-13 Hz) | Regeneration |
| **Visualization** | Alpha suppression + theta bursts | Spell casting |
| **Breathing Rhythm** | Rhythmic alpha oscillation pattern | Energy charge |
| **Drowsy** | Rising theta, falling beta | Vulnerability / debuff |
| **Stressed / Anxious** | High beta, low alpha, high blink rate | Rage / berserker |

## How It Works

The classifier uses band-power ratios from your HALO's 4 channels (Fp1, Fp2, O1, O2)
as features. It is a lightweight scikit-learn model (Random Forest) that runs in <1ms
per inference — suitable for the 10 Hz real-time loop.

Features extracted per window:
- Band powers: delta, theta, alpha, beta, gamma × 4 channels = 20
- Ratios: beta/alpha (frontal), theta/alpha (frontal), alpha asymmetry = 3
- Coherence: fronto-occipital alpha coherence = 1
- Temporal: alpha variability (breathing rhythm proxy) = 1
- Total: 25 features per classification window
