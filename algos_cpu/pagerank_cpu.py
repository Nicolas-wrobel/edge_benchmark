import numpy as np
from scipy.sparse import csr_matrix

def _pagerank_sync(adj: csr_matrix, push=True, iters=20, alpha=0.85, **_):
    n = adj.shape[0]
    pr = np.ones(n) / n
    out_deg = np.diff(adj.indptr)
    print(f"[PR][SYNC] START n={n}, iters={iters}, push={push}")
    for i in range(iters):
        if i % 1 == 0:
            print(f"[PR][SYNC] iter {i+1}/{iters}")
        new_pr = np.zeros(n)
        if push:
            for u in range(n):
                if out_deg[u] == 0: continue
                neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
                new_pr[neighbors] += alpha * pr[u] / out_deg[u]
        else:  # Pull = prod adj.T
            for v in range(n):
                incoming = adj[:, v].nonzero()[0]
                s = pr[incoming] / out_deg[incoming]
                new_pr[v] = alpha * s.sum()
        new_pr += (1 - alpha) / n
        pr = new_pr
    print(f"[PR][SYNC] DONE. iters={iters}, edges={int(adj.nnz) * iters}")
    return {"iterations": iters, "edges": int(adj.nnz) * iters}

def _pagerank_async(adj: csr_matrix, push=True, alpha=0.85, eps=1e-4, max_active=None, **_):
    n = adj.shape[0]
    pr = np.zeros(n)
    residu = np.ones(n) / n
    out_deg = np.diff(adj.indptr)
    worklist = list(range(n))
    iters = 0
    edges = 0
    print(f"[PR][ASYNC] START n={n}, push={push}")
    while worklist:
        if iters % 10000 == 0:
            print(f"[PR][ASYNC] it={iters}, worklist={len(worklist)}, edges={edges}")
        u = worklist.pop()
        r = residu[u]
        if r < eps: continue
        pr[u] += r
        residu[u] = 0.0
        if out_deg[u] == 0: continue
        share = alpha * r / out_deg[u]
        neighbors = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
        for v in neighbors:
            prev = residu[v]
            residu[v] += share
            edges += 1
            if prev < eps and residu[v] >= eps:
                worklist.append(v)
        iters += 1
        if max_active and iters >= max_active:
            break
    print(f"[PR][ASYNC] DONE. iters={iters}, edges={edges}")
    return {"iterations": iters, "edges": edges}


pagerank_sync_topo_push   = lambda adj, **kw: _pagerank_sync(adj, push=True, **kw)
pagerank_sync_topo_pull   = lambda adj, **kw: _pagerank_sync(adj, push=False, **kw)
pagerank_sync_data_push   = pagerank_sync_topo_push
pagerank_sync_data_pull   = pagerank_sync_topo_pull
pagerank_async_topo_push  = lambda adj, **kw: _pagerank_async(adj, push=True, **kw)
pagerank_async_topo_pull  = lambda adj, **kw: _pagerank_async(adj, push=False, **kw)
pagerank_async_data_push  = pagerank_async_topo_push
pagerank_async_data_pull  = pagerank_async_topo_pull

for name in [
    "pagerank_sync_topo_push", "pagerank_sync_topo_pull", "pagerank_sync_data_push", "pagerank_sync_data_pull",
    "pagerank_async_topo_push", "pagerank_async_topo_pull", "pagerank_async_data_push", "pagerank_async_data_pull"]:
    locals()[name].__name__ = name
