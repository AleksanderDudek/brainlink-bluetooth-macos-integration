"""
macOS-native BLE adapter for BrainAccess HALO.

Uses ``bleak`` (CoreBluetooth backend on macOS) to communicate directly
with the HALO headband over Bluetooth Low Energy, bypassing the official
SDK which only supports Windows/Linux.

BLE UUIDs were extracted from the official ``libbacore.so`` native library.
"""

import asyncio
import logging
import threading
from typing import Optional

import numpy as np

try:
    from bleak import BleakClient, BleakScanner
    from bleak.backends.characteristic import BleakGATTCharacteristic
    HAS_BLEAK = True
except ImportError:
    BleakGATTCharacteristic = None  # type: ignore[misc,assignment]
    HAS_BLEAK = False

from .constants import CHANNELS, SAMPLE_RATE

log = logging.getLogger("brainlink.ble")

# ─── Known BLE UUIDs (extracted from libbacore.so + real device discovery) ───
# Standard services
SVC_DEVICE_INFO = "0000180a-0000-1000-8000-00805f9b34fb"
SVC_BATTERY     = "0000180f-0000-1000-8000-00805f9b34fb"

# Standard characteristics
CHR_BATTERY_LEVEL   = "00002a19-0000-1000-8000-00805f9b34fb"
CHR_MODEL_NUMBER    = "00002a24-0000-1000-8000-00805f9b34fb"
CHR_SERIAL_NUMBER   = "00002a25-0000-1000-8000-00805f9b34fb"
CHR_FIRMWARE_REV    = "00002a26-0000-1000-8000-00805f9b34fb"
CHR_HARDWARE_REV    = "00002a27-0000-1000-8000-00805f9b34fb"
CHR_MANUFACTURER    = "00002a29-0000-1000-8000-00805f9b34fb"

# ─── BrainAccess custom service (reverse-engineered from libbacore.so) ────────
# All streaming operations live in this single control service.
SVC_CONTROL        = "ace20bb6-5d39-45de-9f21-b101acf45b9b"
CHR_STREAM_CMD     = "49beba02-5f6b-4ca2-b3c9-2e6b71aa9392"  # write — start[0x01]/stop[0x00]
CHR_STREAM_DATA    = "a12f7c4b-9e8d-6a53-1e0f-c274815b36d7"  # notify — EEG data
CHR_STREAM_CONFIG  = "3467e3a2-4511-ae66-7515-b26fce099337"  # write — channel config
CHR_STREAM_STATUS  = "e7945b26-141c-4459-8090-33b8bc85cee8"  # notify — status

# Other services (not used for streaming)
SVC_DATA           = "a3543662-77ab-47e4-a9e1-7c2cff80798f"
SVC_SETTINGS       = "12af3c98-4bd2-71e0-a615-5933cb7d8244"

# ─── Stream config byte format ────────────────────────────────────────────────
# [impedance_mode, gain_ch0..3, enable_mask, rate]
# GainMode: X1=0 X2=1 X4=2 X6=3 X8=4 X12=5  |  StreamRate: 250 Hz = 6
STREAM_CONFIG = bytes([0x00, 0x04, 0x04, 0x04, 0x04, 0x0F, 0x06])

# Packet geometry
RAW_CHANNELS_PER_SAMPLE = 6   # 6 × int8 delta values per 6-byte record
PACKET_FOOTER_LEN = 6          # [enable_mask(1B), counter_u32_LE(4B), unknown(1B)]


class HALOBleAdapter:
    """
    Cross-platform BLE adapter for BrainAccess HALO.
    Works on macOS via bleak's CoreBluetooth backend.

    Workflow::

        adapter = HALOBleAdapter()
        adapter.scan()              # find nearby devices
        adapter.connect("BA HALO 001")
        adapter.start_stream()
        raw = adapter.generate(25)  # drain 25 samples per channel
        adapter.stop_stream()
        adapter.disconnect()
    """

    def __init__(self):
        if not HAS_BLEAK:
            raise RuntimeError("bleak is not installed. Run: pip install bleak")
        self._client: Optional[BleakClient] = None
        self._connected = False
        self._streaming = False
        self._device_address: Optional[str] = None
        self._device_name: Optional[str] = None
        self._sample_rate = SAMPLE_RATE
        self._lock = threading.Lock()
        self._buffer: dict[str, list[float]] = {ch: [] for ch in CHANNELS}
        self._battery_level: Optional[int] = None
        self._model: str = "HALO"
        self._firmware: str = "unknown"
        self._hardware: str = "unknown"
        self._serial: str = "unknown"
        self._manufacturer: str = "unknown"
        self._services_map: dict = {}
        # Delta-decoding accumulators (one per raw channel; bytes 2-5 → EEG)
        self._accum = [0.0] * RAW_CHANNELS_PER_SAMPLE
        # Background asyncio event loop for bleak calls
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ── Internal async helpers ─────────────────────────────────────
    def _ensure_loop(self):
        if self._loop is not None and self._loop.is_running():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    # ── Scanning ───────────────────────────────────────────────────
    def scan(self, timeout: float = 5.0) -> list[dict]:
        """Return a list of nearby BrainAccess devices."""
        return self._run(self._async_scan(timeout))

    async def _async_scan(self, timeout: float) -> list[dict]:
        devices = await BleakScanner.discover(timeout=timeout)
        return [
            {"name": d.name, "address": d.address, "rssi": getattr(d, "rssi", None)}
            for d in devices
            if d.name and any(kw in d.name.upper() for kw in ("HALO", "BRAINACCESS", "BA "))
        ]

    # ── Connection ─────────────────────────────────────────────────
    def connect(self, address_or_name: str) -> dict:
        """Connect by BLE address or device name."""
        return self._run(self._async_connect(address_or_name))

    async def _async_connect(self, address_or_name: str) -> dict:
        address = address_or_name
        if not self._looks_like_address(address_or_name):
            devices = await BleakScanner.discover(timeout=5.0)
            found = next(
                (d for d in devices if d.name and address_or_name.lower() in d.name.lower()),
                None,
            )
            if not found:
                raise RuntimeError(
                    f"Device '{address_or_name}' not found. "
                    "Make sure it's powered on and in range."
                )
            address = found.address
            self._device_name = found.name

        self._client = BleakClient(address)
        await self._client.connect()
        self._connected = True
        self._device_address = address
        await self._discover_services()
        await self._read_device_info()

        return {
            "status": "connected",
            "device": self._device_name or address,
            "address": address,
            "model": self._model,
            "firmware": self._firmware,
            "hardware": self._hardware,
            "serial": self._serial,
            "manufacturer": self._manufacturer,
            "battery_level": self._battery_level,
            "sample_rate": self._sample_rate,
            "eeg_channels": len(CHANNELS),
            "services": self._services_map,
        }

    async def _discover_services(self):
        self._services_map = {
            service.uuid.lower(): [
                {"uuid": ch.uuid.lower(), "properties": ch.properties}
                for ch in service.characteristics
            ]
            for service in self._client.services
        }
        log.info("Discovered %d services", len(self._services_map))

    async def _read_device_info(self):
        for uuid, attr in {
            CHR_MODEL_NUMBER: "_model",
            CHR_FIRMWARE_REV: "_firmware",
            CHR_HARDWARE_REV: "_hardware",
            CHR_SERIAL_NUMBER: "_serial",
            CHR_MANUFACTURER: "_manufacturer",
        }.items():
            try:
                data = await self._client.read_gatt_char(uuid)
                setattr(self, attr, data.decode("utf-8", errors="replace").strip("\x00"))
            except Exception:
                pass
        try:
            self._battery_level = (await self._client.read_gatt_char(CHR_BATTERY_LEVEL))[0]
        except Exception:
            pass

    def disconnect(self):
        if self._streaming:
            self.stop_stream()
        if self._client and self._connected:
            try:
                self._run(self._client.disconnect())
            except Exception:
                pass
        self._connected = False
        self._client = None
        self._device_address = None

    # ── Streaming ──────────────────────────────────────────────────
    def start_stream(self):
        if not self._connected:
            raise RuntimeError("Not connected")
        self._run(self._async_start_stream())
        self._streaming = True

    async def _async_start_stream(self):
        self._accum = [0.0] * RAW_CHANNELS_PER_SAMPLE
        await self._client.write_gatt_char(CHR_STREAM_CONFIG, STREAM_CONFIG, response=True)
        await self._client.start_notify(CHR_STREAM_DATA, self._on_notify)
        await self._client.write_gatt_char(CHR_STREAM_CMD, bytes([0x01]), response=True)
        log.info("HALO BLE stream started")

    def stop_stream(self):
        if self._client and self._streaming:
            try:
                self._run(self._async_stop_stream())
            except Exception:
                pass
        self._streaming = False

    async def _async_stop_stream(self):
        try:
            await self._client.write_gatt_char(CHR_STREAM_CMD, bytes([0x00]), response=True)
        except Exception as e:
            log.warning("Could not send stop command: %s", e)
        try:
            await self._client.stop_notify(CHR_STREAM_DATA)
        except Exception:
            pass

    # ── Packet parsing ─────────────────────────────────────────────
    def _on_notify(self, sender: BleakGATTCharacteristic, data: bytearray):
        with self._lock:
            self._parse_eeg_packet(data)

    def _parse_eeg_packet(self, data: bytearray):
        """
        Parse a delta-encoded EEG notification.

        Layout::

            body   = N × 6 bytes  (N = (len − FOOTER) // 6)
            footer = 6 bytes  [enable_mask | counter_u32_LE | unknown]

        Each 6-byte record holds 6 × int8 deltas; cumulative sum gives the
        raw signal.  Bytes 2–5 map to EEG channels Fp1, Fp2, O1, O2.
        """
        body_len = len(data) - PACKET_FOOTER_LEN
        if body_len < RAW_CHANNELS_PER_SAMPLE or body_len % RAW_CHANNELS_PER_SAMPLE != 0:
            if len(data) != PACKET_FOOTER_LEN:
                log.warning("Unexpected packet size: %d bytes", len(data))
            return

        ch_map = [2, 3, 4, 5]  # raw-record byte indices → Fp1, Fp2, O1, O2
        for r in range(body_len // RAW_CHANNELS_PER_SAMPLE):
            off = r * RAW_CHANNELS_PER_SAMPLE
            for i in range(RAW_CHANNELS_PER_SAMPLE):
                b = data[off + i]
                self._accum[i] += b if b < 128 else b - 256  # uint8 → int8
            for eeg_i, raw_i in enumerate(ch_map):
                self._buffer[CHANNELS[eeg_i]].append(self._accum[raw_i])

        # Cap buffers at 5 seconds
        cap = self._sample_rate * 5
        for ch in CHANNELS:
            if len(self._buffer[ch]) > cap:
                self._buffer[ch] = self._buffer[ch][-cap:]

    # ── Data interface (same API as EEGSimulator) ──────────────────
    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        """Drain up to ``n_samples`` from each channel buffer."""
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
        return self._connected and self._client is not None

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def battery_level(self) -> Optional[int]:
        if self._connected and self._client:
            try:
                self._battery_level = (
                    self._run(self._client.read_gatt_char(CHR_BATTERY_LEVEL))
                )[0]
            except Exception:
                pass
        return self._battery_level

    @property
    def device_info(self) -> dict:
        if not self._connected:
            return {}
        return {
            "device_name": self._device_name,
            "address": self._device_address,
            "model": self._model,
            "firmware": self._firmware,
            "sample_rate": self._sample_rate,
            "eeg_channels": len(CHANNELS),
            "battery_level": self._battery_level,
        }

    def close(self):
        """Full teardown including the background event loop."""
        self.disconnect()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None

    @staticmethod
    def _looks_like_address(s: str) -> bool:
        """Return True if *s* looks like a BLE MAC or macOS UUID string."""
        return (len(s) == 17 and s.count(":") == 5) or (len(s) == 36 and s.count("-") == 4)
