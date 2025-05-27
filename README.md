# Edge-Benchmark

Ce dépôt contient un banc d’essai CPU / GPU / Hybride pour trois algorithmes de graphes (BFS, SSSP, PageRank) inspirés de l’article “Analysis of Parallel Graph Applications” (ICPADS 2024).

### Prérequis

- Python ≥ 3.9  
- Pour le GPU : CUDA ≥ 11.4 et cuDNN installés (JetPack 4.6.4 testé)  
- PyTorch & PyTorch-Geometric :  
  ```bash
  pip install torch==2.1.0+cu118 torchvision torchaudio -f \
        https://download.pytorch.org/whl/torch_stable.html
  pip install pyg-lib torch-scatter torch-sparse torch-cluster \
        torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
  ```

## Installation rapide

```bash
git clone <repo>
cd edge_benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Lancer un bench simple

### Sur Jetson nano

```bash
python3 main.py --graph datasets/CA-GrQc.edgelist --algorithm bfs --backend gpu
```
### Sur PC

```bash
python main.py --graph datasets/CA-GrQc.edgelist --algorithm bfs --backend gpu
```

* `--graph`   chemin vers `.edgelist`, `.mtx`, ou `.txt`
* `--algorithm` `bfs | sssp | pagerank`
* `--backend`   `cpu | gpu | hyb`
* `--source`    id du sommet source (sinon max‑degré)
* `--timeout`   force le timeout (s) d’un test

## Benchmarks exhaustifs

```bash
python3 main.py --benchmark-all --backend gpu \
                --export-csv results_gpu.csv --profile --timeout 200
```

*Par défaut parcourt le dossier `datasets/` et exporte un CSV (puis XLSX) avec toutes les combinaisons (sync/async, push/pull, topo/data).*

### Options utiles

* `--energy` log conso énergie Jetson (JetsonPowerSampler)
* `--profile` génère un fichier `.pstats` par test
* `--force-mp-timeout` utilise systématiquement le timeout multiprocess (évite SIGALRM)

### Mode « hyb »

`--backend hyb` : reste sur CPU tant que la frontier < 5 % des sommets, bascule GPU sinon (et retour CPU quand < 5 %).

## Lire les résultats

Un fichier NoteBook.ipynb se situe à la racine du projet, il permet de visualiser tous les graphes et statistiques sur les benchmarks.

## Structure du dépôt

```
algos_cpu/      implémentations numpy/scipy
algos_gpu/      kernels PyTorch Geometric
algos_hybrid/   CPU↔GPU « frontier threshold »
datasets/       graphes de test (.edgelist)
utils/          chargeurs, métriques, sampling CPU/énergie
main.py         lanceur principal CLI
Notebook.ipynb  fichier notebook avec statistiques
```
