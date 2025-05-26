import numpy as np
from scipy.sparse import csr_matrix

def _pick_source(adj, source=None):
    if source is not None:
        return source
    out_deg = np.diff(adj.indptr)
    return np.argmax(out_deg)

def _sssp_sync(adj: csr_matrix, push=True, source=None, **_):
    n = adj.shape[0]
    src = _pick_source(adj, source)
    dist = np.full(n, np.inf, dtype=np.float32)
    dist[src] = 0
    level = 0
    edges = 0

    print(f"[SSSP][SYNC] START push={push} n={n} src={src}")
    if push:
        frontier = np.zeros(n, dtype=bool)
        frontier[src] = True
        while np.any(frontier):
            print(f"[SSSP][SYNC][push] level={level}, frontier={np.sum(frontier)}, edges={edges}")
            next_frontier = np.zeros(n, dtype=bool)
            idxs = np.where(frontier)[0]
            for u in idxs:
                neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
                for v in neighbors:
                    edges += 1
                    if dist[v] > dist[u] + 1:
                        dist[v] = dist[u] + 1
                        next_frontier[v] = True
            frontier = next_frontier
            level += 1
    else:  # Pull
        active = [src]
        while active:
            print(f"[SSSP][SYNC][pull] level={level}, active={len(active)}, edges={edges}")
            next_active = []
            for v in range(n):
                if dist[v] == np.inf:
                    neighbors = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
                    for u in neighbors:
                        if dist[u] + 1 < dist[v]:
                            dist[v] = dist[u] + 1
                            next_active.append(v)
                            edges += 1
                            break
            if not next_active:
                break
            active = next_active
            level += 1
    finite = np.isfinite(dist).sum()
    max_depth = int(np.nanmax(dist[dist < np.inf])) if finite else 0
    unreachable = n - finite
    print(f"[SSSP][SYNC] DONE. levels={level}, edges={edges}, reached={finite}, unreachable={unreachable}")
    return {'iterations': level, 'unreachable': int(unreachable), 'edges': edges}


def _sssp_async(adj: csr_matrix, push=True, source=None, **_):
    n = adj.shape[0]
    src = _pick_source(adj, source)
    dist = np.full(n, np.inf, dtype=np.float32)
    dist[src] = 0
    edges = 0
    iters = 0

    print(f"[SSSP][ASYNC] START push={push} n={n} src={src}")
    if push:
        queue = [src]
        while queue:
            if iters % 1000 == 0:
                print(f"[SSSP][ASYNC][push] it={iters}, queue={len(queue)}, edges={edges}")
            u = queue.pop(0)
            neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
            for v in neighbors:
                edges += 1
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
            iters += 1
    else:  # Pull
        active = [src]
        while active:
            print(f"[SSSP][ASYNC][pull] it={iters}, active={len(active)}, edges={edges}")
            next_active = []
            for v in range(n):
                if dist[v] == np.inf:
                    neighbors = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
                    for u in neighbors:
                        if dist[u] < np.inf:
                            dist[v] = dist[u] + 1
                            next_active.append(v)
                            edges += 1
                            break
            if not next_active:
                break
            active = next_active
            iters += 1
    finite = np.isfinite(dist).sum()
    max_depth = int(np.nanmax(dist[dist < np.inf])) if finite else 0
    unreachable = n - finite
    print(f"[SSSP][ASYNC] DONE. iters={iters}, edges={edges}, reached={finite}, unreachable={unreachable}")
    return {'iterations': iters, 'unreachable': int(unreachable), 'edges': edges}


# --- Alias pour chaque mode ---
sssp_sync_topo_push   = lambda adj, **kw: _sssp_sync(adj, push=True, **kw)
sssp_sync_topo_pull   = lambda adj, **kw: _sssp_sync(adj, push=False, **kw)
sssp_sync_data_push   = sssp_sync_topo_push
sssp_sync_data_pull   = sssp_sync_topo_pull
sssp_async_topo_push  = lambda adj, **kw: _sssp_async(adj, push=True, **kw)
sssp_async_topo_pull  = lambda adj, **kw: _sssp_async(adj, push=False, **kw)
sssp_async_data_push  = sssp_async_topo_push
sssp_async_data_pull  = sssp_async_topo_pull

for name in [
    "sssp_sync_topo_push", "sssp_sync_topo_pull", "sssp_sync_data_push", "sssp_sync_data_pull",
    "sssp_async_topo_push", "sssp_async_topo_pull", "sssp_async_data_push", "sssp_async_data_pull"]:
    locals()[name].__name__ = name
