"""
BrainAccess SDK adapter for the HALO headband.

Requires the ``brainaccess`` package (Windows / Linux only).  The server
falls back to the BLE adapter (:mod:`brainlink.halo_ble`) on macOS.
"""

import logging
import threading
from typing import Optional

import numpy as np

from .constants import CHANNELS, HALO_ELECTRODE_MAP, SAMPLE_RATE

try:
    import brainaccess.core as ba_core
    from brainaccess.core.eeg_manager import EEGManager
    import brainaccess.core.eeg_channel as eeg_channel
    from brainaccess.core.gain_mode import GainMode
    HAS_BRAINACCESS = True
except ImportError:
    HAS_BRAINACCESS = False

log = logging.getLogger("brainlink.sdk")


class BrainAccessHALO:
    """
    Real-time reader for the BrainAccess HALO via the official SDK.

    Provides the same ``generate(n_samples)`` interface as
    :class:`~.simulator.EEGSimulator` so it can be swapped in transparently.
    """

    def __init__(self):
        if not HAS_BRAINACCESS:
            raise RuntimeError("brainaccess package is not installed")
        self._mgr: Optional[EEGManager] = None
        self._connected = False
        self._streaming = False
        self._device_name: Optional[str] = None
        self._sample_rate = SAMPLE_RATE
        self._eeg_channel_count = 0
        self._channel_indices: dict[str, int] = {}
        self._lock = threading.Lock()
        self._buffer: dict[str, list[float]] = {ch: [] for ch in CHANNELS}
        self._battery_level: Optional[int] = None
        self._core_initialized = False

    # ── Discovery ──────────────────────────────────────────────────
    def scan(self) -> list[dict]:
        if not self._core_initialized:
            ba_core.init()
            self._core_initialized = True
        return [{"name": d.name} for d in ba_core.scan()]

    # ── Connection ─────────────────────────────────────────────────
    def connect(self, device_name: str) -> dict:
        """Connect to a HALO device by its BLE name."""
        if self._connected:
            return {"status": "already_connected", "device": self._device_name}

        if not self._core_initialized:
            ba_core.init()
            self._core_initialized = True

        self._mgr = EEGManager()
        status = self._mgr.connect(device_name)
        if status == 2:
            self._mgr.destroy()
            self._mgr = None
            raise RuntimeError("Stream incompatible — update HALO firmware")
        if status != 0:
            self._mgr.destroy()
            self._mgr = None
            raise RuntimeError(f"Connection failed (status={status})")

        self._connected = True
        self._device_name = device_name

        device_features = self._mgr.get_device_features()
        self._eeg_channel_count = device_features.electrode_count()

        enabled = 0
        for ch_name, idx in HALO_ELECTRODE_MAP.items():
            if idx < self._eeg_channel_count:
                ch_id = eeg_channel.ELECTRODE_MEASUREMENT + idx
                self._mgr.set_channel_enabled(ch_id, True)
                self._mgr.set_channel_gain(ch_id, GainMode.X8)
                enabled += 1

        self._mgr.set_channel_enabled(eeg_channel.SAMPLE_NUMBER, True)
        self._mgr.load_config()
        self._sample_rate = self._mgr.get_sample_frequency()

        for ch_name, idx in HALO_ELECTRODE_MAP.items():
            if idx < self._eeg_channel_count:
                ch_id = eeg_channel.ELECTRODE_MEASUREMENT + idx
                self._channel_indices[ch_name] = self._mgr.get_channel_index(ch_id)

        self._mgr.set_callback_chunk(self._on_chunk)

        try:
            bi = self._mgr.get_battery_info()
            self._battery_level = bi.level
        except Exception:
            pass

        info = self._mgr.get_device_info()
        return {
            "status": "connected",
            "device": device_name,
            "eeg_channels": self._eeg_channel_count,
            "sample_rate": self._sample_rate,
            "model": info.device_model.name,
            "firmware": (
                f"{info.firmware_version.major}."
                f"{info.firmware_version.minor}."
                f"{info.firmware_version.patch}"
            ),
        }

    def disconnect(self):
        if self._streaming:
            self.stop_stream()
        if self._mgr:
            self._mgr.destroy()
            self._mgr = None
        self._connected = False
        self._device_name = None
        self._channel_indices.clear()

    def close(self):
        self.disconnect()
        if self._core_initialized:
            ba_core.close()
            self._core_initialized = False

    # ── Streaming ──────────────────────────────────────────────────
    def start_stream(self):
        if not self._connected or not self._mgr:
            raise RuntimeError("Not connected")
        self._mgr.start_stream()
        self._streaming = True

    def stop_stream(self):
        if self._mgr and self._streaming:
            self._mgr.stop_stream()
        self._streaming = False

    def _on_chunk(self, chunk_arrays: list, chunk_size: int):
        """SDK data callback — runs on a native thread."""
        with self._lock:
            for ch_name, data_idx in self._channel_indices.items():
                samples = chunk_arrays[data_idx][:chunk_size].tolist()
                self._buffer[ch_name].extend(samples)
                max_len = self._sample_rate * 5
                if len(self._buffer[ch_name]) > max_len:
                    self._buffer[ch_name] = self._buffer[ch_name][-max_len:]

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        """Drain up to ``n_samples`` from the internal buffer (same API as EEGSimulator)."""
        data = {}
        with self._lock:
            for ch in CHANNELS:
                buf = self._buffer[ch]
                take = min(n_samples, len(buf))
                data[ch] = np.array(buf[:take]) if take > 0 else np.zeros(0)
                self._buffer[ch] = buf[take:]
        return data

    # ── Status ─────────────────────────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def battery_level(self) -> Optional[int]:
        if self._connected and self._mgr:
            try:
                self._battery_level = self._mgr.get_battery_info().level
            except Exception:
                pass
        return self._battery_level

    @property
    def device_info(self) -> dict:
        if not self._connected:
            return {}
        return {
            "device_name": self._device_name,
            "sample_rate": self._sample_rate,
            "eeg_channels": self._eeg_channel_count,
            "battery_level": self.battery_level,
        }
