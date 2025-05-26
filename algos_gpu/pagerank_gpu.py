import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data
from utils.torch_graph import to_pyg_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _gpu_pagerank_sync(G, push=True, iters=20, alpha=0.85, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n = data.num_nodes
    src, dst = (data.edge_index if push else data.edge_index[[1, 0]])
    print(f"[GPU][PR][SYNC][{'push' if push else 'pull'}] Start: n={n}, edges={src.numel()}")
    outdeg = torch.bincount(src, minlength=n).clamp(min=1).float()
    pr = torch.full((n,), 1./n, device=DEVICE)
    for i in range(iters):
        if i % 5 == 0 or i < 3:
            print(f"[GPU][PR][SYNC][{'push' if push else 'pull'}] it {i+1:4d}/{iters}")
        contrib = pr[src] / outdeg[src]
        pr.zero_()
        scatter_add(contrib, dst, out=pr)
        pr.mul_(alpha).add_((1-alpha)/n)
    print(f"[GPU][PR][SYNC][{'push' if push else 'pull'}] Done: iters={iters}, edges={src.numel() * iters}")
    return {"iterations": iters, "edges": src.numel() * iters}

def _gpu_pagerank_async(G, push=True, alpha=0.85, eps=1e-4, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n = data.num_nodes
    src, dst = data.edge_index
    outdeg = torch.bincount(src, minlength=n).clamp(min=1).float()
    pr = torch.zeros(n, device=DEVICE)
    resid = torch.full((n,), 1./n, device=DEVICE)
    active = torch.ones(n, dtype=torch.bool, device=DEVICE)
    iters, walked = 0, 0
    while active.any():
        u = active.nonzero(as_tuple=False).flatten()
        active.zero_()
        share = alpha * resid[u] / outdeg[u]
        walked += src.numel()
        resid.index_fill_(0, u, 0.)
        pr[u] += share * outdeg[u] / alpha
        if push:
            valid = dst < resid.size(0)
            dst_valid = dst[valid]
            share_valid = share[valid]
            resid.scatter_add_(0, dst_valid, share_valid.repeat_interleave(outdeg[u][valid].long()))
        else:
            valid = src < resid.size(0)
            src_valid = src[valid]
            share_valid = share[valid]
            resid.scatter_add_(0, src_valid, share_valid.repeat_interleave(outdeg[u].long()))
        active = resid > eps
        iters += 1
    print(f"[GPU][PR][ASYNC][{'push' if push else 'pull'}] Done: iters={iters}, edges={walked}")
    return {"iterations": iters, "edges": walked}

# --- ALIAS PAR MODE ---
def pagerank_sync_topo_push(G, **kw):  return _gpu_pagerank_sync(G, push=True,  **kw)
def pagerank_sync_topo_pull(G, **kw):  return _gpu_pagerank_sync(G, push=False, **kw)
def pagerank_sync_data_push(G, **kw):  return _gpu_pagerank_sync(G, push=True,  **kw)
def pagerank_sync_data_pull(G, **kw):  return _gpu_pagerank_sync(G, push=False, **kw)
def pagerank_async_topo_push(G, **kw): return _gpu_pagerank_async(G, push=True,  **kw)
def pagerank_async_topo_pull(G, **kw): return _gpu_pagerank_async(G, push=False, **kw)
def pagerank_async_data_push(G, **kw): return _gpu_pagerank_async(G, push=True,  **kw)
def pagerank_async_data_pull(G, **kw): return _gpu_pagerank_async(G, push=False, **kw)

pagerank_sync_topo_push.__name__ = "pagerank_sync_topo_push"
pagerank_sync_topo_pull.__name__ = "pagerank_sync_topo_pull"
pagerank_sync_data_push.__name__ = "pagerank_sync_data_push"
pagerank_sync_data_pull.__name__ = "pagerank_sync_data_pull"
pagerank_async_topo_push.__name__ = "pagerank_async_topo_push"
pagerank_async_topo_pull.__name__ = "pagerank_async_topo_pull"
pagerank_async_data_push.__name__ = "pagerank_async_data_push"
pagerank_async_data_pull.__name__ = "pagerank_async_data_pull"
