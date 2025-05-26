from typing import Optional, List
import psutil, threading, time

class CPUUsageSampler:
    def __init__(self, interval: float = 0.05):
        self.proc     = psutil.Process()
        self.interval = interval
        self._stop    = threading.Event()
        self.values: List[float] = []
        self.thread   = None

    def _sample(self):
        self.proc.cpu_percent(None)          # amorçage
        while not self._stop.is_set():
            self.values.append(self.proc.cpu_percent(self.interval))

    def start(self):
        self._stop.clear()
        self.values = []
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join()

    def mean(self) -> Optional[float]:
        return round(sum(self.values) / len(self.values), 1) if self.values else None
