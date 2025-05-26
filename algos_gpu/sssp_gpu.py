import torch, warnings
from torch_geometric.data import Data
from utils.torch_graph import to_pyg_data
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _sssp_gpu_core(G, push=True, source=None, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n = data.num_nodes
    src, dst = (data.edge_index if push else data.edge_index[[1,0]])
    if source is None and hasattr(data, "default_source_id"):
        true_source_id = data.default_source_id
    elif source is not None:
        true_source_id = source
    else:
        true_source_id = 0
    if true_source_id not in data.nid_map:
        print(f"[GPU][SSSP] ERREUR: source {true_source_id} n’existe pas.")
        raise ValueError("Source SSSP non trouvée dans nid_map")
    src_id = data.nid_map[true_source_id]

    print(f"[GPU][SSSP][{'push' if push else 'pull'}] Start: n={n}, edges={src.numel()}")
    print(f"[GPU][SSSP][{'push' if push else 'pull'}] Source ID {true_source_id} → idx {src_id}")

    dist = torch.full((n,), 1e9, dtype=torch.int32, device=DEVICE)
    dist[src_id] = 0

    changed, iters, walked = True, 0, 0
    while changed:
        if iters % 50 == 0 or iters < 10:
            print(f"[GPU][SSSP][{'push' if push else 'pull'}]   it {iters:4d}")
        relax = dist[src] + 1
        better = relax < dist[dst]
        changed = better.any().item()
        dist[dst[better]] = relax[better]
        walked += src.numel()
        iters += 1

    print(f"[GPU][SSSP][{'push' if push else 'pull'}] Terminé: iterations={iters}, edges={walked}, nodes visited={(dist < 1e9).sum().item()}")
    return {"iterations": iters, "edges": walked}

def sssp_delta_push(G, *, delta=4, source=None, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n          = data.num_nodes
    src, dst   = data.edge_index

    src_id = 0 if source is None else data.nid_map.get(source, 0)

    dist    = torch.full((n,), float('inf'), device=DEVICE)
    dist[src_id] = 0.

    bucket = [[src_id]]
    walked, iters = 0, 0

    while bucket and any(bucket):
        while bucket and not bucket[0]:
            bucket.pop(0)
        if not bucket: break
        S = bucket[0]; bucket[0] = []

        active = torch.tensor(S, device=DEVICE)
        improved = True
        while improved:
            mask = (src.unsqueeze(1) == active).any(1)
            neigh = dst[mask]; walked += mask.sum().item()
            relax = dist[src[mask]] + 1
            better = relax < dist[neigh]
            if better.any():
                dist[neigh[better]] = relax[better]
                active = neigh[better]
                improved = True
            else:
                improved = False
        idx = (dist[S] / delta).long()
        for v, b in zip(S, idx):
            while len(bucket) <= b: bucket.append([])
            bucket[b].append(v)
        iters += 1

    print(f"[GPU][SSSP][DELTA][push] Terminé: iterations={iters}, edges={walked}")
    return {"iterations": iters, "edges": walked}

# --- ALIAS PAR MODE (GPU classique) ---
def sssp_sync_topo_push(G, **kw):  return _sssp_gpu_core(G, push=True,  **kw)
def sssp_sync_topo_pull(G, **kw):  return _sssp_gpu_core(G, push=False, **kw)
def sssp_sync_data_push(G, **kw):  return _sssp_gpu_core(G, push=True,  **kw)
def sssp_sync_data_pull(G, **kw):  return _sssp_gpu_core(G, push=False, **kw)
def sssp_async_topo_push(G, **kw): return _sssp_gpu_core(G, push=True,  **kw)
def sssp_async_topo_pull(G, **kw): return _sssp_gpu_core(G, push=False, **kw)
def sssp_async_data_push(G, **kw): return _sssp_gpu_core(G, push=True,  **kw)
def sssp_async_data_pull(G, **kw): return _sssp_gpu_core(G, push=False, **kw)

sssp_sync_topo_push.__name__ = "sssp_sync_topo_push"
sssp_sync_topo_pull.__name__ = "sssp_sync_topo_pull"
sssp_sync_data_push.__name__ = "sssp_sync_data_push"
sssp_sync_data_pull.__name__ = "sssp_sync_data_pull"
sssp_async_topo_push.__name__ = "sssp_async_topo_push"
sssp_async_topo_pull.__name__ = "sssp_async_topo_pull"
sssp_async_data_push.__name__ = "sssp_async_data_push"
sssp_async_data_pull.__name__ = "sssp_async_data_pull"

# --- Alias spécifique DELTA ---
sssp_delta_push.__name__ = "sssp_delta_push"
