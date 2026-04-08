"""DSP pipeline: ring-buffer management, band-power extraction, blink detection."""

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, welch

from .constants import BANDS, CHANNELS, SAMPLE_RATE, WINDOW_SAMPLES


class DSPPipeline:
    """
    Maintains per-channel ring buffers and computes spectral features.

    Usage::

        dsp = DSPPipeline()
        dsp.push(raw_data)           # raw_data: {ch: np.ndarray}
        if dsp.has_enough():
            powers = dsp.compute_band_powers()
            blinks = dsp.detect_blinks()
    """

    # Exponential MA time constant for DC/drift removal.
    # τ = 0.5 s → HP cutoff ≈ 0.3 Hz; converges to 95 % in ~1.5 s.
    # (Previous τ=2 s was too slow: electrodes still drifting after 6 s.)
    _HP_ALPHA: float = float(np.exp(-1.0 / (SAMPLE_RATE * 0.5)))

    def __init__(self):
        self.buffers: dict[str, np.ndarray] = {ch: np.array([]) for ch in CHANNELS}
        # Running low-pass state used for HP = signal − LP (DC baseline).
        # Initialised to None; seeded from the first sample of each channel.
        self._hp_state: dict[str, float | None] = {ch: None for ch in CHANNELS}
        # 50 Hz notch filter (Europe/Poland mains) applied before buffering.
        # Quality factor Q=30 gives ~1.7 Hz notch width.
        b_notch, a_notch = iirnotch(50.0, Q=30, fs=SAMPLE_RATE)
        # Convert to SOS for numerical stability
        from scipy.signal import tf2sos
        self._notch_sos = tf2sos(b_notch, a_notch)
        self._notch_state: dict[str, np.ndarray | None] = {ch: None for ch in CHANNELS}
        # Per-band SOS bandpass filters
        self._filters = {
            band: butter(4, [lo, hi], btype="band", fs=SAMPLE_RATE, output="sos")
            for band, (lo, hi) in BANDS.items()
        }

    def push(self, data: dict[str, np.ndarray]):
        """DC-remove + 50 Hz notch + buffer each channel (ring, max 2 s)."""
        a = self._HP_ALPHA
        for ch in CHANNELS:
            raw = data[ch].astype(float)
            if len(raw) == 0:
                continue

            # ── Step 1: remove DC / slow electrode drift (HP via LP subtraction)
            if self._hp_state[ch] is None:
                # Seed at median of first chunk for robustness against
                # single spike outliers at stream start.
                self._hp_state[ch] = float(np.median(raw))
            lp = self._hp_state[ch]
            hp = np.empty_like(raw)
            for i, x in enumerate(raw):
                lp = a * lp + (1.0 - a) * x
                hp[i] = x - lp
            self._hp_state[ch] = lp

            # ── Step 2: 50 Hz notch (mains interference)
            if self._notch_state[ch] is None:
                # zi shape: (n_sections, 2) — one initial condition per section
                from scipy.signal import sosfilt_zi
                self._notch_state[ch] = sosfilt_zi(self._notch_sos) * hp[0]
            notched, self._notch_state[ch] = sosfilt(
                self._notch_sos, hp, zi=self._notch_state[ch]
            )

            self.buffers[ch] = np.concatenate([self.buffers[ch], notched])
            if len(self.buffers[ch]) > SAMPLE_RATE * 2:
                self.buffers[ch] = self.buffers[ch][-SAMPLE_RATE * 2:]

    def has_enough(self) -> bool:
        """Return True when all channels have at least one full FFT window."""
        return all(len(self.buffers[ch]) >= WINDOW_SAMPLES for ch in CHANNELS)

    def compute_band_powers(self) -> dict[str, dict[str, float]]:
        """Return per-channel band power estimates {channel: {band: µV²}}."""
        result = {}
        for ch in CHANNELS:
            buf = self.buffers[ch][-WINDOW_SAMPLES:]
            freqs, psd = welch(buf, fs=SAMPLE_RATE, nperseg=min(128, len(buf)))
            result[ch] = {
                band: float(np.mean(psd[(freqs >= lo) & (freqs <= hi)]) or 0.0)
                for band, (lo, hi) in BANDS.items()
            }
        return result

    def detect_blinks(self) -> int:
        """Count blink artifacts in the last second using frontal channels."""
        count = 0
        for ch in ("Fp1", "Fp2"):
            buf = self.buffers[ch][-WINDOW_SAMPLES:]
            threshold = np.std(buf) * 4
            peaks = np.where(np.abs(buf) > threshold)[0]
            if len(peaks) > 0:
                count += 1 + int(np.sum(np.diff(peaks) > 20))
        return count
