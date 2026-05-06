# DenStream

A Python implementation of the **DenStream** algorithm for density-based clustering over evolving data streams. This project provides a robust, decoupled Online-Offline architecture capable of discovering clusters of arbitrary shapes while handling noise and concept drift.
based on the [DenStream Paper](DenStreamPaper.pdf).



## Installation
1. Clone the repository and navigate into the directory:
```bash
   git clone --recurse-submodules https://gitlab2.cip.ifi.lmu.de/dbs_sep/dbs_streamclustering26/block1/denstream.git
   cd denstream
   ```
2. (Optional but recommended) Create and activate a virtual environment:
```bash
    python -m venv .venv
    source .venv/bin/activat
   ```
   
3. Install required dependencies
    ```pip install -r requirements.txt```

## Usage
You can run the algorithm in two ways: via the command-line evaluation script, or by importing it directly into your own Python code/Jupyter Notebook

### Option 1: Command Line Interface
Use ```run_denstream.py``` to evaluate the algorithm on standard datasets
Example run (Complex9 dataset):
```bash
    python run_denstream.py --dataset complex9 --epsilon 0.04 --mu 2.0 --beta 0.55 --lambda_decay 0.0001 --offline_interval 1
   ```

#### Available Arguments: 

```--dataset```: The dataset to use (e.g., complex9, rbf3, har)

```--epsilon```: Micro-cluster radius threshold

```--mu```: Minimum weight for a core micro-cluster

```--beta```: Outlier weight threshold factor (Note: beta * mu must be > 1)

```--lambda_decay```: Fading function decay factor

```--offline_interval```: Run offline clustering and evaluation every N batches

### Option 2: Python / Jupyter Notebook

You can easily import DenStream into your own scripts or use the provided [playground.ipynb](playground.ipynb) for live visualization
```python
import numpy as np
from src.denstream import DenStream

# 1. Initialize the algorithm
denstream = DenStream(epsilon=0.04, mu=2.0, beta=0.55, lambda_decay=0.0001)

# 2. Feed data in batches (Online Phase)
# Example data: batch (2D array), timestamps (1D array)
denstream.learn(batch, timestamps)

# 3. Retrieve Macro-Clusters on demand (Offline Phase)
# Returns a dictionary mapping Micro-Cluster IDs to Macro-Cluster Labels
clustering_dict = denstream.predict()
```


## Project Structure

```src/```: Core algorithm implementation

- ```denstream.py```: Main orchestrator

-    ```online_phase.py```: Routing, updating, and pruning Micro-Clusters

-    ```offline_phase.py```: offline clustering using DBSCAN

-    ```micro_clusters.py```: The core Data Structure and decay mathematics

```run_denstream.py```: CLI script for batch processing and evaluation (CMM / TSIndex)

```playground.ipynb```: Interactive notebook for animating and visualizing the stream

```baseframework/```: Underlying utilities for data loading and metrics evaluation



