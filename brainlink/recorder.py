"""Session recorder: streams raw EEG samples to CSV + JSON metadata."""

import csv
import json
import threading
import time
import uuid
from typing import Optional

from .constants import CHANNELS, RECORDINGS_DIR, SAMPLE_RATE

RECORDINGS_DIR.mkdir(exist_ok=True)


class SessionRecorder:
    """
    Records raw EEG channel data to disk during live sessions.

    Output per session:
    - ``<id>_data.csv`` — timestamped sample rows (one row per sample)
    - ``<id>_meta.json`` — session metadata (name, duration, sample rate, …)
    """

    def __init__(self):
        self.recording = False
        self.session_id: Optional[str] = None
        self.name: str = ""
        self.duration_sec: int = 300
        self.start_time: float = 0.0
        self.notes: str = ""
        self._source: str = "unknown"
        self._csv_file = None
        self._csv_writer = None
        self._sample_count = 0
        self._lock = threading.Lock()

    def start(self, name: str, duration_sec: int, source: str = "unknown") -> dict:
        if self.recording:
            raise RuntimeError("Already recording")
        self.session_id = uuid.uuid4().hex[:12]
        self.name = name
        self.duration_sec = duration_sec
        self.start_time = time.time()
        self.notes = ""
        self._sample_count = 0
        self._source = source

        csv_path = RECORDINGS_DIR / f"{self.session_id}_data.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp", "sample_idx"] + CHANNELS)
        self.recording = True
        return self.info()

    def stop(self) -> dict:
        info = self.info()
        self.recording = False
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self._save_meta()
        return info

    def add_samples(self, raw_data: dict) -> bool:
        """
        Append a chunk of raw samples.  Returns ``True`` if the recording was
        auto-stopped because the requested duration was reached.
        """
        if not self.recording or not self._csv_writer:
            return False
        if time.time() - self.start_time >= self.duration_sec:
            self.stop()
            return True
        ts = round(time.time(), 4)
        with self._lock:
            n = min(len(v) for v in raw_data.values()) if raw_data else 0
            for i in range(n):
                row = [ts, self._sample_count]
                for ch in CHANNELS:
                    arr = raw_data.get(ch)
                    row.append(round(float(arr[i]), 4) if arr is not None and i < len(arr) else 0.0)
                self._csv_writer.writerow(row)
                self._sample_count += 1
            if self._csv_file:
                self._csv_file.flush()
        return False

    def info(self) -> dict:
        elapsed = time.time() - self.start_time if self.recording else 0
        return {
            "session_id": self.session_id,
            "name": self.name,
            "recording": self.recording,
            "duration_sec": self.duration_sec,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": round(max(0, self.duration_sec - elapsed), 1),
            "sample_count": self._sample_count,
            "notes": self.notes,
        }

    def _save_meta(self):
        if not self.session_id:
            return
        meta = {
            "session_id": self.session_id,
            "name": self.name,
            "duration_sec": self.duration_sec,
            "actual_duration_sec": round(time.time() - self.start_time, 1),
            "start_time": self.start_time,
            "end_time": time.time(),
            "sample_count": self._sample_count,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "notes": self.notes,
            "source": self._source,
        }
        with open(RECORDINGS_DIR / f"{self.session_id}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
