import numpy as np, torch
from torch_geometric.data import Data
from utils.torch_graph import to_pyg_data

_cache = {}

def fast_load_edgelist(path, device="cuda"):
    try:
        if path in _cache:
            return _cache[path]
        src_list, dst_list = [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                u, v = map(int, line.split())
                src_list.append(u)
                dst_list.append(v)
                # Tu peux ajouter un abort ici si tu veux stopper après X edges (optionnel)
        src = torch.tensor(src_list, dtype=torch.long)
        dst = torch.tensor(dst_list, dtype=torch.long)
        data = to_pyg_data(None, device, src=src, dst=dst)
        _cache[path] = data
        return data
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise MemoryError(f"[OOM] lors du chargement du graphe {path}")
        raise
