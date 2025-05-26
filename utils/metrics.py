import os, time, psutil, platform, csv

def detect_platform():
    return "Jetson" if "aarch64" in platform.machine() else "PC"

DEFAULT_KEYS = [
    "graph","algo","mode","access","activation",
    "backend","platform","time_s",
    "iterations","unreachable","edges",
    "rss_MB","cpu_pct","avg_watt"
]

def measure_and_log(metrics: dict, csv_path: str):
    import os, csv, psutil
    # Juste compléter la RAM actuelle si tu veux
    proc = psutil.Process(os.getpid())
    metrics.setdefault("rss_MB", round(proc.memory_info().rss / 1e6, 1))

    # --- Write CSV ---
    first = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DEFAULT_KEYS)
        if first:
            w.writeheader()
        w.writerow({k: metrics.get(k, "") for k in DEFAULT_KEYS})

