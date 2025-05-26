# utils/jetson_power.py
import os
import subprocess, re, atexit, threading, time

class JetsonPowerSampler:
    _regex = re.compile(r"POM_5V_IN\s+(\d+)")

    def __init__(self, period=0.5):
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        cmd = ["/usr/bin/tegrastats", "--interval", str(int(period * 1000))]
        # si besoin sudo
        if os.geteuid() != 0:
            cmd.insert(0, "sudo")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True)
        threading.Thread(target=self._reader, daemon=True).start()
        atexit.register(self.stop)

    def _reader(self):
        for line in self._proc.stdout:
            if self._stop.is_set(): break
            m = self._regex.search(line)
            if m:
                self.samples.append(int(m.group(1)))
        self._proc.stdout.close()

    def stop(self):
        self._stop.set()
        if self._proc.poll() is None:
            self._proc.terminate()

    def avg_watt(self):
        return round(sum(self.samples)/len(self.samples)/1000, 2) if self.samples else None
