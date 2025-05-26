import time

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

from utils.torch_graph import to_pyg_data

_conv_cache = {}                # path  →  Data (CUDA)

def load_graph_gpu(path: str):
    if path not in _conv_cache:
        _conv_cache[path] = to_pyg_data(load_csr_graph(path))
    return _conv_cache[path]


def load_csr_graph(path):
    print(f"[LOAD] Start loading graph: {path}")

    t0 = time.time()
    if path.endswith('.mtx'):
        print("[LOAD] Format: Matrix Market (.mtx)")
        mat = mmread(path)
        print(f"[LOAD] .mtx loaded in {time.time()-t0:.2f}s")
        if not hasattr(mat, 'tocsr'):
            mat = csr_matrix(mat)
        print(f"[LOAD] Converted to CSR in {time.time()-t0:.2f}s")
        return mat.tocsr()
    elif path.endswith('.edgelist') or path.endswith('.txt'):
        print("[LOAD] Format: .edgelist/.txt (src dst par ligne)")
        t1 = time.time()
        edges = np.loadtxt(path, dtype=np.int64)
        print(f"[LOAD] edges loaded: shape={edges.shape} in {time.time()-t1:.2f}s")

        if edges.ndim == 1:
            edges = edges.reshape(1, 2)
        t2 = time.time()
        nodes = np.unique(edges)
        print(f"[LOAD] np.unique done: n_nodes={len(nodes)} in {time.time()-t2:.2f}s")

        t3 = time.time()
        id_map = {int(n): i for i, n in enumerate(nodes)}
        print(f"[LOAD] id_map (id -> idx) created in {time.time()-t3:.2f}s")

        t4 = time.time()
        src = np.array([id_map[int(u)] for u in edges[:, 0]], dtype=np.int32)
        dst = np.array([id_map[int(v)] for v in edges[:, 1]], dtype=np.int32)
        print(f"[LOAD] src/dst remapping done in {time.time()-t4:.2f}s")

        t5 = time.time()
        n = len(nodes)
        data = np.ones(len(src), dtype=np.int8)
        csr = csr_matrix((data, (src, dst)), shape=(n, n))
        print(f"[LOAD] csr_matrix created in {time.time()-t5:.2f}s")

        # Optionnel : propriétés
        csr.default_source_id = int(nodes[0])
        csr.node_ids = nodes

        print(f"[LOAD] TOTAL elapsed: {time.time()-t0:.2f}s for {n} nodes, {len(src)} edges")
        return csr
    else:
        raise ValueError("Format non supporté (attendu : .mtx ou .edgelist)")