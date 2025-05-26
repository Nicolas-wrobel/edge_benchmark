# ──────────────────────────────────────────────────────────────
#  BFS « hybride » : on reste côté CPU tant que la frontier est
#  petite, on bascule sur le GPU dès qu’elle dépasse  TH_RATIO %
#  des sommets.  Retour au CPU dès qu’elle re-devient < TH_RATIO.
# ──────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch_geometric.data import Data
from utils.torch_graph import to_pyg_data          # déjà dispo dans ton dépôt

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TH_RATIO   = 0.05      # ← seuil « 5 % des sommets »
INF_CPU    = 2**31-1   # pour éviter les np.inf dans du int32


# ----------------------------------------------------------------
def _pick_source(n, out_deg, user_src=None):
    if user_src is not None:
        return user_src
    return int(np.argmax(out_deg))                 # plus haut degré sortant


# ----------------------------------------------------------------
def bfs_sync_topo_push(adj, source=None, **_):
    """
    * `adj`  :  soit une CSR SciPy (CPU) soit un Data PyG (déjà sur GPU)
    * Retour dict {iterations, edges, unreachable}
    """
    # --------  Structures CPU  ----------------------------------
    if isinstance(adj, csr_matrix):
        csr   = adj
        pyg   = None
        n     = csr.shape[0]
        deg   = np.diff(csr.indptr)
    else:                                         # déjà un Data GPU
        pyg   = adj
        n     = pyg.num_nodes
        deg   = np.bincount(pyg.edge_index[0].cpu().numpy(),
                            minlength=n)

    src = _pick_source(n, deg, source)
    dist_cpu     = np.full(n, INF_CPU,  dtype=np.int32)
    dist_cpu[src] = 0
    frontier_cpu = np.zeros(n, dtype=bool); frontier_cpu[src] = True

    # --------  Structures GPU  ----------------------------------
    pyg_gpu   = None        # créé la 1ʳᵉ fois qu’on en a besoin
    dist_gpu  = None
    level, edges = 0, 0

    # ============================================================
    while frontier_cpu.any():
        ratio = frontier_cpu.sum() / n

        # ---------- bascule GPU ---------------------------------
        if ratio >= TH_RATIO and DEVICE == "cuda":        # phase GPU
            if pyg_gpu is None:                           # ➊ conversion unique
                pyg_gpu = (adj if isinstance(adj, Data)
                           else to_pyg_data(adj, DEVICE))
            if dist_gpu is None:                          # première phase GPU
                dist_gpu = torch.full((n,), INF_CPU,
                                      dtype=torch.int32, device=DEVICE)
                dist_gpu.copy_(torch.as_tensor(dist_cpu))

            f_idx = torch.from_numpy(np.where(frontier_cpu)[0]).to(DEVICE)
            src_e, dst_e = pyg_gpu.edge_index
            mask      = (src_e[..., None] == f_idx).any(-1)
            neigh     = dst_e[mask]
            edges    += int(mask.sum().cpu())

            unvisited = dist_gpu[neigh] == INF_CPU
            to_act    = neigh[unvisited].unique()
            if to_act.numel() == 0:
                break                                       # plus de nouveau nœud

            dist_gpu[to_act] = level + 1                    # relax
            frontier_cpu     = np.zeros(n, dtype=bool)
            frontier_cpu[to_act.cpu().numpy()] = True       # retourne CPU

            # on propage aussi sur dist_cpu pour être cohérent
            dist_cpu[to_act.cpu().numpy()] = level + 1

        # ---------- phase CPU -----------------------------------
        else:
            idxs  = np.where(frontier_cpu)[0]
            next_frontier = np.zeros(n, dtype=bool)
            for u in idxs:
                if isinstance(adj, csr_matrix):
                    nei = adj.indices[adj.indptr[u]:adj.indptr[u+1]]
                else:                                       # Data PyG mais CPU
                    mask = (adj.edge_index[0] == u)
                    nei  = adj.edge_index[1][mask].cpu().numpy()
                edges += len(nei)
                for v in nei:
                    if dist_cpu[v] == INF_CPU:
                        dist_cpu[v]  = level + 1
                        next_frontier[v] = True
            frontier_cpu = next_frontier

        level += 1                   # ↑ niveau terminé

    unreachable = int((dist_cpu == INF_CPU).sum())
    return {"iterations": level,
            "edges":       int(edges),
            "unreachable": unreachable}


# ----------------------------------------------------------------
#  Alias pour que le nom reflète la combinaison « sync/topo/push »
# ----------------------------------------------------------------
bfs_sync_topo_push.__name__   = "bfs_sync_topo_push"
bfs_sync_topo_pull            = bfs_sync_topo_push   # (pas impl. spécifique)
bfs_sync_data_push            = bfs_sync_topo_push
bfs_sync_data_pull            = bfs_sync_topo_push
bfs_async_topo_push           = bfs_sync_topo_push   # (idem : simplification)
bfs_async_topo_pull           = bfs_sync_topo_push
bfs_async_data_push           = bfs_sync_topo_push
bfs_async_data_pull           = bfs_sync_topo_push
