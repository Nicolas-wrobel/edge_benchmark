import numpy as np
from scipy.sparse import csr_matrix

def _pick_source(adj, source=None):
    if source is not None:
        return source
    # Par défaut : le noeud de plus haut degré sortant
    out_deg = np.diff(adj.indptr)
    return np.argmax(out_deg)

# --- BFS SYNC ---
def _bfs_sync(adj: csr_matrix, push=True, source=None, **_):
    n = adj.shape[0]
    src = _pick_source(adj, source)
    dist = np.full(n, np.inf, dtype=np.float32)
    dist[src] = 0
    frontier = np.zeros(n, dtype=bool)
    frontier[src] = True
    level, edges, reached = 0, 0, 1

    print(f"[BFS][SYNC] START push={push} n={n} src={src}")
    while np.any(frontier):
        print(f"[BFS][SYNC] LEVEL {level} (frontier size={np.sum(frontier)})")
        next_frontier = np.zeros(n, dtype=bool)
        idxs = np.where(frontier)[0]
        if push:
            for u in idxs:
                neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
                edges += len(neighbors)
                for v in neighbors:
                    if dist[v] == np.inf:
                        dist[v] = level + 1
                        next_frontier[v] = True
                        reached += 1
        else:  # Pull
            for v in range(n):
                if dist[v] == np.inf:
                    neighbors = adj.indices[adj.indptr[v]:adj.indptr[v+1]]
                    for u in neighbors:
                        if dist[u] < np.inf:
                            dist[v] = level + 1
                            next_frontier[v] = True
                            reached += 1
                            break
        frontier = next_frontier
        if level % 1 == 0:  # Affiche à chaque level (tu peux moduler !)
            print(f"[BFS][SYNC] Reached={reached} edges={edges}")
        level += 1
    print(f"[BFS][SYNC] DONE. levels={level}, edges={edges}, reached={reached}")
    return {"iterations": level, "edges": edges, "reached": reached}



# --- BFS ASYNC ---
def _bfs_async(adj: csr_matrix, push=True, source=None, **_):
    n = adj.shape[0]
    src = _pick_source(adj, source)
    dist = np.full(n, np.inf, dtype=np.float32)
    dist[src] = 0
    edges = 0
    iters = 0

    print(f"[BFS][ASYNC] START push={push} n={n} src={src}")
    if push:
        queue = [src]
        while queue:
            if iters % 1000 == 0:
                print(f"[BFS][ASYNC][push] it={iters}, queue={len(queue)}, edges={edges}")
            u = queue.pop(0)
            neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
            for v in neighbors:
                edges += 1
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    queue.append(v)
            iters += 1
    else:  # Pull
        active = [src]
        while active:
            print(f"[BFS][ASYNC][pull] it={iters}, active={len(active)}, edges={edges}")
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
    print(f"[BFS][ASYNC] DONE. iters={iters}, edges={edges}")
    return {"iterations": iters, "edges": edges}


# --- Alias pour chaque mode ---
bfs_sync_topo_push   = lambda adj, **kw: _bfs_sync(adj, push=True, **kw)
bfs_sync_topo_pull   = lambda adj, **kw: _bfs_sync(adj, push=False, **kw)
bfs_sync_data_push   = bfs_sync_topo_push
bfs_sync_data_pull   = bfs_sync_topo_pull
bfs_async_topo_push  = lambda adj, **kw: _bfs_async(adj, push=True, **kw)
bfs_async_topo_pull  = lambda adj, **kw: _bfs_async(adj, push=False, **kw)
bfs_async_data_push  = bfs_async_topo_push
bfs_async_data_pull  = bfs_async_topo_pull

for name in [
    "bfs_sync_topo_push", "bfs_sync_topo_pull", "bfs_sync_data_push", "bfs_sync_data_pull",
    "bfs_async_topo_push", "bfs_async_topo_pull", "bfs_async_data_push", "bfs_async_data_pull"]:
    locals()[name].__name__ = name
