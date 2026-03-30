# BrainLink — EEG-to-Game Real-time Engine

A self-contained desktop application that processes BrainAccess HALO EEG data
(or simulated data) and exposes game-ready brain metrics in real time.

```
brainlink-desktop/
├── run.py              ← Main launcher (creates venv, installs deps, starts)
├── START.bat           ← Double-click on Windows
├── start.sh            ← Double-click on Mac/Linux
├── server.py           ← FastAPI server + EEG pipeline + simulator
├── dashboard/
│   └── index.html      ← Real-time monitoring dashboard
├── requirements.txt
└── README.md
```

## Quick Start

### Option A: Double-click
- **Windows**: double-click `START.bat`
- **Mac/Linux**: double-click `start.sh` (or `chmod +x start.sh && ./start.sh`)

### Option B: Command line
```bash
python run.py
```

That's it. The launcher will:
1. Create a Python virtual environment (`.venv/`)
2. Install all dependencies
3. Start the server on port 8420
4. Open your browser to the dashboard

### Options
```bash
python run.py --port 9000     # custom port
python run.py --no-open       # don't auto-open browser
python run.py --install       # only install deps
```

## Requirements

- Python 3.10+ (check with `python --version`)
- No other dependencies needed — everything is installed automatically

## What You Get

### Real-time Dashboard (http://localhost:8420)

The dashboard shows:
- **6 game metrics** as animated bars: Focus, Calm, Valence, Instability, Stress, Flow
- **4 EEG wave traces** (Fp1, Fp2, O1, O2) updating in real time
- **Band power visualization** per channel (delta, theta, alpha, beta, gamma)
- **Combo detector** showing active combo states (super_flow, panic, zen_heal, berserker)
- **Calibration panel** with step-by-step guided flow
- **Settings panel** with live sliders for sensitivity, dead zone, smoothing per metric
- **Simulator controls** with 6 manual states + 4 auto-scenarios

### EEG Simulator

Since you don't have a HALO device yet, the built-in simulator generates
realistic EEG patterns for each mental state:

| State         | What it simulates                              |
|---------------|------------------------------------------------|
| Baseline      | Resting, eyes open — neutral brain activity    |
| Eyes closed   | Massive alpha boost on occipital channels      |
| Focus         | High beta on frontal, suppressed alpha         |
| Stress        | Noisy high-beta everywhere, blink artifacts    |
| Panic spike   | Extreme stress with theta bursts               |
| Flow          | Synchronized alpha+beta across all channels    |

**Auto-scenarios** simulate realistic gaming sessions:
- **Gaming session** — baseline → focus → stress → flow → relax (65s loop)
- **Horror game** — escalating stress with panic spikes (68s loop)
- **Meditation trainer** — deep relaxation → focus → flow (69s loop)
- **Crafting focus** — sustained focus with flow transitions (74s loop)

### REST API (http://localhost:8420/docs)

Full Swagger docs at `/docs`. Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/calibration/start` | Start calibration session |
| `POST /api/v1/calibration/{id}/step/{step}/complete` | Complete calibration step |
| `GET /api/v1/users/{id}/settings` | Get current settings |
| `PATCH /api/v1/users/{id}/settings` | Update settings (partial merge) |
| `WS /api/v1/stream` | WebSocket real-time stream (10 Hz) |
| `POST /api/v1/debug/set-state/{state}` | Change simulator state |
| `GET /api/v1/debug/scenarios` | List auto-scenarios |

### WebSocket Frame Format

Every 100ms the server pushes:
```json
{
  "ts": 1710684600123,
  "metrics": {
    "focus": 0.72, "calm": 0.15, "valence": 0.61,
    "instability": 0.08, "stress": 0.34, "flow": 0.45
  },
  "active_combos": ["super_flow"],
  "combo_timers": {"super_flow": {"active_sec": 2.1}},
  "band_powers": {"Fp1": {"delta":0.4,"theta":0.9,"alpha":5.3,"beta":3.8,"gamma":0.1}, ...},
  "raw_preview": {"Fp1": [-2.3, 1.1, 0.8, -0.5], ...},
  "blink_count": 0,
  "simulator": {"state": "focus", "auto_mode": false}
}
```

## Switching to Real HALO

When your device arrives, replace `EEGSimulator` in `server.py` with:

```python
from brainaccess.core import BrainAccessDevice

class HALOReader:
    def __init__(self):
        self.device = BrainAccessDevice()
        self.device.connect()
        self.state = "live"

    def generate(self, n_samples):
        return self.device.read(n_samples)

    # ... keep set_state etc as no-ops
```

The rest of the pipeline (DSP, Feature Engine, API, Dashboard) stays identical.
