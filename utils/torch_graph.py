import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

import torch
from torch_geometric.data import Data

def to_pyg_data(G=None, device="cuda", *, src=None, dst=None):
    if isinstance(G, csr_matrix):
        # indices et indptr CPU
        indptr, indices = G.indptr, G.indices
        src = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64),
                        np.diff(indptr))
        dst = indices.astype(np.int64)
        src = torch.from_numpy(src)
        dst = torch.from_numpy(dst)
        return to_pyg_data(None, device, src=src, dst=dst)  # recursion

    if G is None:
        unique_nodes = torch.cat([src, dst]).unique(sorted=True)
        id2idx = {int(id_.item()): i for i, id_ in enumerate(unique_nodes)}
        src_idx = torch.tensor([id2idx[int(s.item())] for s in src], dtype=torch.long)
        dst_idx = torch.tensor([id2idx[int(d.item())] for d in dst], dtype=torch.long)
        n = len(unique_nodes)
        data = Data(edge_index=torch.stack([src_idx, dst_idx], 0), num_nodes=n)
        data.nid_map = {int(k): int(v) for k, v in id2idx.items()}
        # ---- NEW: set default source as node with highest out-degree
        # (compute on src)
        src_np = src.cpu().numpy()
        counts = dict()
        for node in src_np:
            counts[node] = counts.get(node, 0) + 1
        data.default_source_id = max(counts, key=counts.get)
    else:
        node_list = list(G.nodes())
        id2idx = {int(v): i for i, v in enumerate(node_list)}
        edge_index = torch.tensor(
            [[id2idx[int(u)], id2idx[int(v)]] for u, v in G.edges()], dtype=torch.long
        ).T
        n = len(node_list)
        data = Data(edge_index=edge_index, num_nodes=n)
        data.nid_map = id2idx
        # ---- NEW: compute default source on G (highest out-degree)
        degrees = dict(G.out_degree())
        data.default_source_id = max(degrees, key=degrees.get)

    s, d = data.edge_index
    data.edge_index_rev = torch.stack([d, s], 0)

    return data.to(device)

