#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Edge-Benchmark – lanceur principal
#  • Bench CPU / GPU (Jetson ou PC x86)
#  • Time-out adaptatif (SIGALRM sous Linux, fallback multiprocess ailleurs)
#  • Profilage optionnel (.pstats)
#  • Journalisation CSV (+ export XLSX)
# ─────────────────────────────────────────────────────────────────────────────
import argparse, importlib, os, platform, re, sys, time, multiprocessing
import psutil, torch, cProfile

from utils.cpu_sampler         import CPUUsageSampler
from utils.graph_loader_fast   import fast_load_edgelist
from utils.graph_loader        import load_csr_graph, load_graph_gpu
from utils.metrics             import measure_and_log, detect_platform
from utils.jetson_power        import JetsonPowerSampler
from typing                    import Optional   

# ──────────────────────────  Détection signal.alarm  ─────────────────────────
try:
    import signal
    USE_SIGNAL = hasattr(signal, "SIGALRM")
except ImportError:
    USE_SIGNAL = False

# Sur certaines distributions Jetson, alarm() hors thread principal lève EPERM
if detect_platform() == "Jetson":
    multiprocessing.set_start_method("spawn", force=True)
    USE_SIGNAL = False         # → on utilisera toujours le fallback multiproc

# Utilitaire : appel « sûr » à signal.alarm
def safe_alarm(seconds: int = 0):
    if USE_SIGNAL:
        try:
            signal.alarm(seconds)
        except (OSError, AttributeError):   # EPERM ou absence de signal
            pass

# ───────────────────────────  Paramètres globaux  ────────────────────────────
DEFAULT_TIMEOUT, MEDIUM_TIMEOUT, LARGE_TIMEOUT = 60, 180, 480        # s
DEFAULT_EDGE_TH, MEDIUM_EDGE_TH, LARGE_EDGE_TH = 50_000, 1_000_000, 5_000_000

ALGO_MODULES = {
    "bfs"      : {"cpu": "algos_cpu.bfs_cpu",
                  "gpu": "algos_gpu.bfs_gpu",
                  "hyb": "algos_hybrid.bfs_hybrid"},
    "sssp"     : {"cpu": "algos_cpu.sssp_cpu",
                  "gpu": "algos_gpu.sssp_gpu"},
    "pagerank" : {"cpu": "algos_cpu.pagerank_cpu",
                  "gpu": "algos_gpu.pagerank_gpu"},
}

PRIORITIZED = {
    ("bfs",      "sync",  "pull", "topo"),
    ("bfs",      "async", "push", "data"),
    ("sssp",     "sync",  "pull", "topo"),
    ("sssp",     "async", "push", "data"),
    ("pagerank", "sync",  "pull", "topo"),
    ("pagerank", "async", "push", "data"),
}

# ────────────────────────────  Fonctions outils  ─────────────────────────────
def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)

def profile_filename(a) -> str:
    g = os.path.basename(a.graph).replace(".", "-")
    return sanitize_filename(
        f"profile_{detect_platform()}_{a.backend}_{g}_"
        f"{a.algorithm}_{a.mode}_{a.access}_{a.activation}.pstats"
    )

def compute_timeout(edge_cnt: int, override: Optional[int] = None) -> int:
    if override is not None:
        return override
    if edge_cnt < MEDIUM_EDGE_TH:
        return DEFAULT_TIMEOUT
    if edge_cnt < LARGE_EDGE_TH:
        return MEDIUM_TIMEOUT
    return LARGE_TIMEOUT

def repeat_bench(fn, *args, min_time=0.3, **kwargs):
    """Exécute la fonction au moins `min_time` secondes, retourne (res, t, n)."""
    n, t0, res = 0, time.perf_counter(), None
    while time.perf_counter() - t0 < min_time:
        res = fn(*args, **kwargs)
        n += 1
    return res, time.perf_counter() - t0, n

# ────────  SECTION  Timeout multiprocess  (remplace la précédente) ────────
import multiprocessing, time, importlib

_MP_MANAGER = None
def _manager():
    global _MP_MANAGER
    if _MP_MANAGER is None:
        _MP_MANAGER = multiprocessing.Manager()
    return _MP_MANAGER

def _bench_entry(ret_dict,  # <- dict à retourner
                 mod_path: str, fn_name: str,
                 graph_path: str, backend: str,
                 src: Optional[int], min_time: float):
    """Exécuté dans le sous-processus.  AUCUN objet non-picklable en entrée."""
    from utils.graph_loader_fast import fast_load_edgelist
    from utils.graph_loader      import load_csr_graph, load_graph_gpu
    from utils.metrics           import detect_platform
    import psutil
    proc = psutil.Process()
    proc.cpu_percent(None)

    # 1) chargement
    if backend == "gpu":
        G = fast_load_edgelist(graph_path) if graph_path.endswith(".edgelist") \
            else load_graph_gpu(graph_path)
        use_gpu = True
    else:
        G = load_csr_graph(graph_path)
        use_gpu = False

    # 2) exécution
    algo_mod = importlib.import_module(mod_path)
    algo_fn  = getattr(algo_mod, fn_name)

    n, t0, res = 0, time.perf_counter(), None
    while time.perf_counter() - t0 < min_time:
        res = algo_fn(G, use_gpu=use_gpu, source=src)
        n += 1
    cpu_pct = proc.cpu_percent(None)  # % moyen sur l’intervalle
    ret_dict["out"] = {
        "res": res, "elapsed": time.perf_counter() - t0,
        "n": n, "cpu": cpu_pct
    }

def run_with_timeout(mod_path, fn_name, graph_path,
                     backend, src, timeout, min_time=0.3):
    """Bench avec timeout ; renvoie (out|None, status)."""
    d = _manager().dict()
    p = multiprocessing.Process(
        target=_bench_entry,
        args=(d, mod_path, fn_name, graph_path, backend, src, min_time)
    )
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return None, "TIMEOUT"
    return d.get("out"), "OK" if "out" in d else "ERROR"

# ─────────────────────────────  run_single  ───────────────────────────────────
def run_single(a, graph, edge_cnt):
    """Lance 1 combinaison (algo+mode+backend) et log le résultat."""
    timeout = compute_timeout(edge_cnt, a.timeout)
    mod     = importlib.import_module(ALGO_MODULES[a.algorithm][a.backend])
    fn_name = f"{a.algorithm}_{a.mode}_{a.activation}_{a.access}"
    algo_fn = getattr(mod, fn_name)

    # ---------- wrappers ----------
    cpu_samp = CPUUsageSampler(0.05)
    def bench():
        cpu_samp.start()
        res, elapsed, n = repeat_bench(algo_fn, graph,
                                       use_gpu=(a.backend == "gpu"),
                                       source=a.source)
        cpu_samp.stop()
        return {"res": res, "elapsed": elapsed, "n": n, "cpu": cpu_samp.mean()}

    # ---------- exécution avec timeout ----------
    # - SIGALRM (Linux) si disponible et pas forcé,
    # - sinon fallback multiprocess.
    try:
        mod_path = ALGO_MODULES[a.algorithm][a.backend]
        fn_name = f"{a.algorithm}_{a.mode}_{a.activation}_{a.access}"
        if a.backend == "hyb":
            # Pas de sous-processus
            out = bench()
            status = "OK"
        elif USE_SIGNAL and not a.force_mp_timeout:
            # ───── Voie SIGALRM (Linux) ───────────────────────────────────
            class _TO(TimeoutError):
                pass

            def _hdl(_, __):
                raise _TO()

            signal.signal(signal.SIGALRM, _hdl);
            safe_alarm(timeout)

            # profilage SEULEMENT dans le parent
            prof = cProfile.Profile() if a.profile else None
            if prof: prof.enable()
            out = bench()
            if prof:
                prof.disable();
                prof.dump_stats(profile_filename(a))

            status = "OK"
        else:  # multiprocess
            out, status = run_with_timeout(
                mod_path, fn_name, a.graph,
                a.backend, a.source, timeout
            )
        safe_alarm(0)
    except TimeoutError:          status = "TIMEOUT"
    except MemoryError:           status = "OOM"
    except BaseException as e:    status, out = "ERROR", None; print(f"[ERROR] {e}")
    finally:                      safe_alarm(0); cpu_samp.stop()

    # ---------- métriques ----------
    metrics = dict(
        graph=a.graph, algo=a.algorithm, mode=a.mode,
        access=a.access, activation=a.activation,
        backend=a.backend, platform=detect_platform(), status=status
    )
    if status == "OK":
        # ─── temps moyen par itération ───────────────────────────────────────────
        time_s = round(out["elapsed"] / out["n"], 6) if out["n"] else None

        # ─── % CPU : None  ⇒  champ vide dans le CSV ─────────────────────────────
        cpu_val = out.get("cpu")
        cpu_pct = round(cpu_val, 1) if cpu_val is not None else None

        metrics.update(time_s=time_s,
                    repeat   = out["n"],
                    cpu_pct  = cpu_pct)
        for k, v in out["res"].items():
            if isinstance(v, (int, float)): metrics[k] = v / out["n"]
            else:                           metrics[k] = v
    else:
        metrics.update(time_s=None, repeat=0, cpu_pct=None)

    measure_and_log(metrics, a.export_csv)
    print("Metrics:", metrics)

# ─────────────────────────────  benchmark_all  ───────────────────────────────
def benchmark_all(a):
    dset_dir = "datasets"
    graphs = ([a.graph] if a.graph else
              [os.path.join(dset_dir, f) for f in sorted(os.listdir(dset_dir))
               if f.lower().endswith((".edgelist", ".mtx", ".txt"))])
    if not graphs:
        sys.exit("[ERROR] Aucun dataset trouvé")

    for g in graphs:
        print(f"[BENCH] {g}")
        # Chargement (GPU d'abord, fallback CPU)
        try:
            if a.backend == "gpu":
                try:
                    G = fast_load_edgelist(g) if g.endswith(".edgelist") else load_graph_gpu(g)
                    edges_cnt = G.edge_index.size(1)
                except (RuntimeError, MemoryError, OSError):
                    print("[FALLBACK] GPU OOM → CPU")
                    G = load_csr_graph(g); edges_cnt = G.nnz; a.backend = "cpu"
            else:
                G = load_csr_graph(g); edges_cnt = G.nnz
        except MemoryError:
            print("[SKIP] OOM au chargement"); continue

        # Boucle algorithmes / combinaisons
        for algo in (["bfs","sssp","pagerank"] if a.algorithm is None else [a.algorithm]):
            for mode in ["sync","async"]:
                if algo == "pagerank" and mode == "async": continue
                for access in ["push","pull"]:
                    if algo == "pagerank" and mode=="sync" and access=="push": continue
                    for act in ["topo","data"]:
                        if edges_cnt > LARGE_EDGE_TH and \
                           (algo,mode,access,act) not in PRIORITIZED:
                            continue
                        comb = argparse.Namespace(**vars(a))
                        comb.algorithm, comb.graph = algo, g
                        comb.mode, comb.access, comb.activation = mode, access, act
                        try: run_single(comb, G, edges_cnt)
                        except Exception as e:
                            print(f"[WARN] {algo}-{mode}-{access}-{act}: {e}")

        # Mémoire
        del G; import gc; gc.collect()
        if a.backend == "gpu": torch.cuda.empty_cache()

# ───────────────────────────────  CLI  ───────────────────────────────────────
def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--graph")
    p.add_argument("--algorithm", choices=["bfs","sssp","pagerank"])
    p.add_argument("--backend",   choices=["cpu", "gpu", "hyb"], default="cpu")
    p.add_argument("--export-csv", default="results.csv")
    p.add_argument("--source", type=int)
    p.add_argument("--energy", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--benchmark-all", action="store_true")
    p.add_argument("--timeout", type=int, help="Override du timeout (s)")
    p.add_argument("--force-mp-timeout", action="store_true",
                   help="Utiliser systématiquement le timeout multiprocess")
    return p.parse_args()

# ───────────────────────────────  Main  ──────────────────────────────────────
if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_cli()
    if args.benchmark_all:
        benchmark_all(args)
        # Export XLSX (optionnel)
        try:
            import pandas as pd
            if os.path.exists(args.export_csv):
                df = pd.read_csv(args.export_csv)
                df.to_excel(args.export_csv.replace(".csv",".xlsx"), index=False)
                print("→ Export XLSX généré")
        except Exception as e:
            print(f"[EXCEL] Échec export : {e}")
    else:
        # exécution d’une seule combinaison
        if args.backend == "gpu":
            G = fast_load_edgelist(args.graph) if args.graph.endswith(".edgelist") else load_graph_gpu(args.graph)
            edges = G.edge_index.size(1)
        else:
            G = load_csr_graph(args.graph); edges = G.nnz
        run_single(args, G, edges)
