import torch
from torch_geometric.data import Data
from utils.torch_graph import to_pyg_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- BFS GPU SYNC ---
def bfs_gpu_sync(G, push=True, data_driven=True, source=None, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n = data.num_nodes
    print(f"[GPU][BFS][SYNC][{'push' if push else 'pull'}][{'data' if data_driven else 'topo'}] n={n}, edges={data.edge_index.shape[1]}")
    # Source mapping
    if source is None and hasattr(data, "default_source_id"):
        true_source_id = data.default_source_id
    elif source is not None:
        true_source_id = source
    else:
        true_source_id = min(data.nid_map.keys())
    if true_source_id not in data.nid_map:
        print(f"[GPU][BFS] ERREUR: source {true_source_id} absent.")
        raise ValueError("ID source non trouvé.")
    src_id = data.nid_map[true_source_id]
    print(f"[GPU][BFS][SYNC] Source ID {true_source_id} → idx {src_id}")

    dist = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    dist[src_id] = 0
    active = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    active[src_id] = True
    iterations = 0
    edges = 0

    while active.any():
        if iterations % 50 == 0 or iterations < 10:
            print(f"[GPU][BFS][SYNC][{'push' if push else 'pull'}]   it {iterations:4d}: active={active.sum().item()}")
        if data_driven:
            # data-driven: BFS classique avec active set
            if push:
                mask = active[data.edge_index[0]]
                neighbors = data.edge_index[1][mask]
                to_activate = neighbors[dist[neighbors] < 0].unique()
                edges += mask.sum().item()
            else:
                mask = active[data.edge_index_rev[0]]
                neighbors = data.edge_index_rev[1][mask]
                to_activate = neighbors[dist[neighbors] < 0].unique()
                edges += mask.sum().item()
        else:
            # topo-driven: niveau par niveau
            current_level = (dist == iterations)
            if push:
                mask = current_level[data.edge_index[0]]
                neighbors = data.edge_index[1][mask]
                to_activate = neighbors[dist[neighbors] < 0].unique()
                edges += mask.sum().item()
            else:
                mask = current_level[data.edge_index_rev[0]]
                neighbors = data.edge_index_rev[1][mask]
                to_activate = neighbors[dist[neighbors] < 0].unique()
                edges += mask.sum().item()
        if to_activate.numel() == 0:
            break
        dist[to_activate] = iterations + 1
        active = torch.zeros_like(active)
        active[to_activate] = True
        iterations += 1
    unreachable = int((dist < 0).sum().item())
    print(f"[GPU][BFS][SYNC][{'push' if push else 'pull'}] Done: iters={iterations}, edges={edges}, reached={(dist >= 0).sum().item()}")
    return {
        "iterations": iterations,
        "edges": edges,
        "unreachable": unreachable
    }

# --- BFS GPU ASYNC ---
def bfs_gpu_async(G, push=True, data_driven=True, source=None, **_):
    data = G if isinstance(G, Data) else to_pyg_data(G, DEVICE)
    n = data.num_nodes
    print(f"[GPU][BFS][ASYNC][{'push' if push else 'pull'}][{'data' if data_driven else 'topo'}] n={n}, edges={data.edge_index.shape[1]}")
    if source is None and hasattr(data, "default_source_id"):
        true_source_id = data.default_source_id
    elif source is not None:
        true_source_id = source
    else:
        true_source_id = min(data.nid_map.keys())
    if true_source_id not in data.nid_map:
        print(f"[GPU][BFS][ASYNC] ERREUR: source {true_source_id} absent.")
        raise ValueError("ID source non trouvé.")
    src_id = data.nid_map[true_source_id]
    print(f"[GPU][BFS][ASYNC] Source ID {true_source_id} → idx {src_id}")

    dist = torch.full((n,), -1, dtype=torch.int32, device=DEVICE)
    dist[src_id] = 0
    # On simule la worklist à la main (moins efficace que SYNC sur GPU, mais même logique)
    q = [src_id]
    edges = 0
    iters = 0

    while q:
        if iters % 50 == 0 or iters < 10:
            print(f"[GPU][BFS][ASYNC][{'push' if push else 'pull'}]   it {iters:4d}: qlen={len(q)}")
        next_q = []
        q_tensor = torch.tensor(q, device=DEVICE)
        if push:
            # Pour chaque noeud actif, ajoute ses successeurs si non visités
            for u in q:
                mask = data.edge_index[0] == u
                neighbors = data.edge_index[1][mask]
                for v in neighbors:
                    edges += 1
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        next_q.append(v.item())
        else:
            # Pull async: pour chaque noeud non atteint, si un prédécesseur atteint, on l'ajoute à la queue
            unvisited = (dist < 0).nonzero(as_tuple=False).flatten().tolist()
            for v in unvisited:
                preds = data.edge_index_rev[1][data.edge_index_rev[0] == v]
                for u in preds:
                    edges += 1
                    if dist[u] >= 0:
                        dist[v] = dist[u] + 1
                        next_q.append(v)
                        break
        q = next_q
        iters += 1
    unreachable = int((dist < 0).sum().item())
    print(f"[GPU][BFS][ASYNC][{'push' if push else 'pull'}] Done: iters={iters}, edges={edges}, reached={(dist >= 0).sum().item()}")
    return {
        "iterations": iters,
        "edges": edges,
        "unreachable": unreachable
    }

# --- ALIAS EXACTS POUR CHAQUE MODE (comme sur CPU) ---
def bfs_sync_topo_push(G, **kw):   return bfs_gpu_sync(G, push=True,  data_driven=False, **kw)
def bfs_sync_topo_pull(G, **kw):   return bfs_gpu_sync(G, push=False, data_driven=False, **kw)
def bfs_sync_data_push(G, **kw):   return bfs_gpu_sync(G, push=True,  data_driven=True,  **kw)
def bfs_sync_data_pull(G, **kw):   return bfs_gpu_sync(G, push=False, data_driven=True,  **kw)

def bfs_async_topo_push(G, **kw):  return bfs_gpu_async(G, push=True,  data_driven=False, **kw)
def bfs_async_topo_pull(G, **kw):  return bfs_gpu_async(G, push=False, data_driven=False, **kw)
def bfs_async_data_push(G, **kw):  return bfs_gpu_async(G, push=True,  data_driven=True,  **kw)
def bfs_async_data_pull(G, **kw):  return bfs_gpu_async(G, push=False, data_driven=True,  **kw)

bfs_sync_topo_push.__name__ = "bfs_sync_topo_push"
bfs_sync_topo_pull.__name__ = "bfs_sync_topo_pull"
bfs_sync_data_push.__name__ = "bfs_sync_data_push"
bfs_sync_data_pull.__name__ = "bfs_sync_data_pull"
bfs_async_topo_push.__name__ = "bfs_async_topo_push"
bfs_async_topo_pull.__name__ = "bfs_async_topo_pull"
bfs_async_data_push.__name__ = "bfs_async_data_push"
bfs_async_data_pull.__name__ = "bfs_async_data_pull"
