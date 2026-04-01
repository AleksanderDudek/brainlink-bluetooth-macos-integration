# BrainLink — EEG-to-Game Real-time Engine

A self-contained desktop application that processes **BrainAccess HALO** EEG
data (or simulated data) and exposes game-ready brain metrics in real time over
WebSocket and a REST API.

```
brainlink-desktop/
├── run.py              ← One-command launcher (creates venv, installs deps, starts server)
├── server.py           ← FastAPI app — routes + background EEG loop
├── brainlink/          ← Core pipeline package
│   ├── constants.py    ← Shared constants (sample rate, bands, channels, …)
│   ├── simulator.py    ← EEG simulator with mental-state scenarios
│   ├── dsp.py          ← Ring buffers + band-power extraction (Welch PSD)
│   ├── features.py     ← Metric engine + combo detector
│   ├── calibration.py  ← Guided calibration session
│   ├── recorder.py     ← CSV + JSON session recorder
│   ├── halo_ble.py     ← macOS BLE adapter (bleak / CoreBluetooth)
│   └── halo_sdk.py     ← Official SDK adapter (Windows / Linux)
├── dashboard/
│   └── index.html      ← Real-time monitoring dashboard
├── recordings/         ← Saved EEG sessions (gitignored)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
python run.py
```

The launcher will:
1. Create a Python virtual environment (`.venv/`)
2. Install all dependencies from `requirements.txt`
3. Start the server on port **8420**
4. Open your browser to the dashboard

```bash
python run.py --port 9000    # custom port
python run.py --no-open      # skip auto-opening browser
python run.py --install      # install deps only
```

## Requirements

- **Python 3.10+**
- No other manual setup required — everything is installed automatically

## What you get

### Real-time dashboard  `http://localhost:8420`

| Panel | Contents |
|---|---|
| Metrics | 6 animated bars: Focus · Calm · Valence · Instability · Stress · Flow |
| EEG traces | 4 channels (Fp1, Fp2, O1, O2) at 10 Hz |
| Band powers | Delta / Theta / Alpha / Beta / Gamma per channel |
| Combos | Active combo states: `super_flow` · `panic` · `zen_heal` · `berserker` |
| Calibration | Step-by-step guided flow |
| Settings | Live sliders for sensitivity, dead-zone, and smoothing per metric |
| Simulator | 6 manual states + 4 auto-scenarios |

### WebSocket stream  `ws://localhost:8420/api/v1/stream`

10 Hz push, JSON frame:

```json
{
  "ts": 1710684600123,
  "metrics": { "focus": 0.72, "calm": 0.15, "valence": 0.61,
               "instability": 0.08, "stress": 0.34, "flow": 0.45 },
  "active_combos": ["super_flow"],
  "combo_timers": { "super_flow": { "active_sec": 2.1 } },
  "band_powers": { "Fp1": { "delta": 0.4, "theta": 0.9, "alpha": 5.3,
                             "beta": 3.8, "gamma": 0.1 } },
  "raw_preview": { "Fp1": [-2.3, 1.1, 0.8, -0.5] },
  "blink_count": 0,
  "source": "simulator",
  "simulator": { "state": "focus", "auto_mode": false }
}
```

### REST API  `http://localhost:8420/docs`

Full Swagger UI at `/docs`.  Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/calibration/start` | Begin guided calibration |
| `POST` | `/api/v1/calibration/{id}/step/{step}/complete` | Advance one calibration step |
| `GET`  | `/api/v1/users/{id}/settings` | Read metric settings |
| `PATCH`| `/api/v1/users/{id}/settings` | Partial-update metric settings |
| `GET`  | `/api/v1/device/status` | Hardware / source status |
| `POST` | `/api/v1/device/scan` | Scan for BLE devices |
| `POST` | `/api/v1/device/connect` | Connect to HALO |
| `POST` | `/api/v1/recording/start` | Start a session recording |
| `POST` | `/api/v1/debug/set-state/{state}` | Change simulator state |

### EEG simulator

| State | What it simulates |
|---|---|
| `baseline` | Resting, eyes open — neutral activity |
| `eyes_closed` | Occipital alpha surge |
| `focus` | High frontal beta, suppressed alpha |
| `stress` | Noisy high-beta, blink artifacts |
| `panic_spike` | Extreme stress with theta bursts |
| `flow` | Fronto-occipital alpha-beta synchrony |

**Auto-scenarios** loop through realistic gaming sessions automatically:
`gaming_session` · `horror_game` · `meditation_trainer` · `crafting_focus`

## Connecting a real HALO device

The server auto-detects which backend is available:

| Platform | Backend | Install |
|---|---|---|
| macOS | `bleak` BLE adapter | already in `requirements.txt` |
| Windows / Linux | BrainAccess SDK | `pip install brainaccess` |

Once a device is connected, switch the source via the dashboard or:

```bash
curl -X POST http://localhost:8420/api/v1/device/connect \
     -H "Content-Type: application/json" \
     -d '{"device_name": "BA HALO 001"}'

curl -X POST http://localhost:8420/api/v1/source/device
```

## Project structure

```
brainlink/
├── constants.py   — sample rate, channel list, frequency bands
├── simulator.py   — EEGSimulator class + AUTO_SCENARIOS
├── dsp.py         — DSPPipeline (ring buffer, Welch PSD, blink detector)
├── features.py    — FeatureEngine, CalibrationProfile, UserSettings
├── calibration.py — CalibrationSession (5-step guided flow)
├── recorder.py    — SessionRecorder (CSV + JSON output)
├── halo_ble.py    — HALOBleAdapter (macOS CoreBluetooth via bleak)
└── halo_sdk.py    — BrainAccessHALO (official SDK, Win/Linux)
```

## License

[GNU Affero General Public License v3.0](LICENSE) — you may use, study, and modify this software freely, but any distributed or network-hosted derivative must also be released under AGPL v3. Commercial use requires the author's written permission.
