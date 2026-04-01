"""DSP pipeline: ring-buffer management, band-power extraction, blink detection."""

import numpy as np
from scipy.signal import butter, welch

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

    def __init__(self):
        self.buffers: dict[str, np.ndarray] = {ch: np.array([]) for ch in CHANNELS}
        # Pre-compute SOS bandpass filters for each frequency band
        self._filters = {
            band: butter(4, [lo, hi], btype="band", fs=SAMPLE_RATE, output="sos")
            for band, (lo, hi) in BANDS.items()
        }

    def push(self, data: dict[str, np.ndarray]):
        """Append samples to each channel buffer, keeping at most 2 seconds."""
        for ch in CHANNELS:
            self.buffers[ch] = np.concatenate([self.buffers[ch], data[ch]])
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
