# EEG Dataset Sources for Mental State Training

Curated list of open EEG datasets suitable for training meditation, relaxation,
focus, visualization, and biofeedback mental state classifiers.

---

## Tier 1 — Highly Recommended (direct mental-state labels)

### 1. DEAP — Dataset for Emotion Analysis using EEG
- **URL**: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- **What**: 32 participants, 32-channel EEG + peripherals, watching music videos
- **Labels**: Valence, arousal, dominance, liking (continuous 1-9)
- **Use for**: Stress vs. calm classification, emotional valence
- **Format**: Python pickle / BDF
- **Channels usable**: Fp1, Fp2, O1, O2 extractable from 32-ch montage
- **Size**: ~5 GB
- **License**: Free for research (requires registration)
- **Paper**: Koelstra et al. (2012), IEEE Trans. Affective Computing

### 2. SEED — SJTU Emotion EEG Dataset
- **URL**: https://bcmi.sjtu.edu.cn/home/seed/
- **What**: 15 participants, 62-channel EEG watching film clips
- **Labels**: Positive, neutral, negative emotion
- **Use for**: Valence/arousal states, calm vs. stressed
- **Format**: MATLAB .mat
- **Size**: ~2 GB
- **License**: Free for research (requires application)

### 3. PhysioNet — EEG Motor Movement / Imagery
- **URL**: https://physionet.org/content/eegmmidb/1.0.0/
- **What**: 109 subjects, 64-channel EEG, motor execution and imagery tasks
- **Labels**: Rest, left fist, right fist, both fists, both feet (imagery + real)
- **Use for**: **Visualization** detection — motor imagery is mental visualization
- **Format**: EDF
- **Size**: ~3.4 GB
- **License**: Open Data Commons (ODC-BY)
- **Why excellent**: Motor imagery is fundamentally visualization; the alpha
  suppression + theta burst patterns during imagery directly map to your
  "visualization" mental state.

### 4. LEMON — Leipzig Study for Mind-Body-Emotion Interactions
- **URL**: https://openneuro.org/datasets/ds000221
- **What**: 228 participants, 62-channel EEG, resting state eyes-open/closed
- **Labels**: Eyes open vs. eyes closed (alpha suppression paradigm)
- **Use for**: **Relaxation/meditation baseline** — eyes-closed resting produces
  the same dominant alpha pattern as beginning meditation
- **Format**: BIDS (BrainVision)
- **Size**: ~60 GB (EEG subset ~8 GB)
- **License**: CC0

### 5. Mind-Wandering EEG
- **URL**: https://openneuro.org/datasets/ds003768
- **What**: EEG during sustained attention task with mind-wandering probes
- **Labels**: On-task (focused) vs. mind-wandering
- **Use for**: **Focus vs. unfocused** classification
- **Format**: BIDS
- **License**: CC0

---

## Tier 2 — Very Useful (indirect but mappable labels)

### 6. Resting-State EEG and Trait Anxiety
- **URL**: https://openneuro.org/datasets/ds007609
- **What**: 51 participants, resting EEG with anxiety trait scores
- **Labels**: Trait anxiety levels mapped to alpha/theta power
- **Use for**: Stress continuum modeling, anxiety-state signatures
- **Format**: BIDS
- **License**: CC0

### 7. DREAMER — Multimodal Emotion Dataset
- **URL**: https://zenodo.org/records/546113
- **What**: 23 participants, 14-channel EEG, watching film clips
- **Labels**: Valence, arousal, dominance
- **Use for**: Emotion → mental state mapping (low arousal+high valence = calm)
- **Format**: MATLAB .mat
- **Size**: ~50 MB (very compact!)
- **License**: CC-BY-4.0
- **Why good**: Only 14 channels — closer to your 4-ch HALO setup

### 8. Mental Arithmetic EEG (PhysioNet)
- **URL**: https://physionet.org/content/eeg-during-mental-arithmetic-tasks/1.0.0/
- **What**: 36 subjects, 19-channel EEG during mental math
- **Labels**: Rest vs. mental arithmetic
- **Use for**: Focused attention / cognitive load detection
- **Format**: EDF
- **License**: ODC-BY

### 9. BCI Competition IV — Dataset 2a
- **URL**: https://www.bbci.de/competition/iv/
- **What**: 9 subjects, 22-channel EEG, 4-class motor imagery
- **Labels**: Left hand, right hand, both feet, tongue imagery
- **Use for**: Visualization sub-types, imagery detection
- **Format**: GDF
- **License**: Free for research

### 10. EEG Pre/Post Intervention Dataset
- **URL**: https://openneuro.org/datasets/ds007558
- **What**: Multi-group EEG recordings before/after intervention
- **Labels**: Pre vs. post intervention (state change)
- **Use for**: Measuring biofeedback-induced state transitions
- **Format**: BIDS
- **License**: CC0

---

## Tier 3 — Supplementary (for augmenting specific states)

### 11. ROAMM — Reading Observed at Mindless Moments
- **URL**: https://openneuro.org/datasets/ds007629
- **What**: EEG + eye-tracking during reading with mind-wandering annotations
- **Labels**: Attentive reading vs. mind-wandering
- **Use for**: Additional focus/unfocus samples

### 12. Kaggle — EEG Brainwave Dataset (Mental State)
- **URL**: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state
- **What**: Muse headband recordings during focused attention and relaxation
- **Labels**: Concentrating, neutral, relaxed
- **Use for**: Direct 3-class mental state pretraining
- **Format**: CSV
- **Size**: ~200 MB
- **License**: CC0
- **Why relevant**: Low-channel-count consumer EEG — very similar to HALO setup

### 13. Kaggle — EEG Data for Mental Attention State Detection
- **URL**: https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection
- **What**: Single-channel EEG during attention tasks
- **Labels**: Focused, unfocused, drowsy
- **Use for**: Drowsiness/alertness gradient

### 14. CHB-MIT Scalp EEG (PhysioNet)
- **URL**: https://physionet.org/content/chbmit/1.0.0/
- **What**: Continuous scalp EEG (pediatric)
- **Labels**: Normal vs. seizure
- **Use for**: Artifact rejection training, abnormal state detection
- **Format**: EDF
- **License**: ODC-BY

---

## Recommended Training Strategy

### Phase 1: Base model (rule-based → ML hybrid)
Use datasets **#4 (LEMON)** + **#12 (Kaggle Mental State)** + **#8 (Mental Arithmetic)**
to train a 4-class classifier: `relaxed`, `focused`, `stressed`, `neutral`.

### Phase 2: Add visualization detection
Add **#3 (PhysioNet Motor Imagery)** + **#9 (BCI Competition)** to detect
`visualization` as alpha suppression + theta bursts during motor imagery.

### Phase 3: Meditation & breathing
Use **#1 (DEAP)** low-arousal high-valence segments + **#7 (DREAMER)** +
your own recorded sessions (via BrainLink's built-in recorder) to train
`meditation` and `breathing_rhythm` states.

### Phase 4: Personal calibration
Fine-tune on the user's own recorded data (BrainLink recordings) to adapt
the model to individual EEG signatures — this is the most impactful step
for a consumer 4-channel device.

---

## Channel Mapping Notes

Your HALO has 4 channels: **Fp1, Fp2, O1, O2**

Most datasets have many more channels. When preprocessing, extract only the
channels closest to your HALO positions:
- **Fp1** (frontal left) — present in nearly all 10-20 montages
- **Fp2** (frontal right) — present in nearly all 10-20 montages
- **O1** (occipital left) — present in most 10-20 montages
- **O2** (occipital right) — present in most 10-20 montages

This 4-channel subset is extractable from every dataset listed above.
