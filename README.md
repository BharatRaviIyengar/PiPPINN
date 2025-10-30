# PiPPINN

PiPPINN is a framework for protein-protein interaction (PPI) prediction using Graph Neural Networks (GNNs). This repository provides a pipeline to encode protein sequences, construct graphs suitable for PyTorch Geometric, and train a custom GNN model to predict interactions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Pipeline](#pipeline)
  - [1. Protein Encoding](#1-protein-encoding)
  - [2. Graph Preparation](#2-graph-preparation)
  - [3. Training the GNN](#3-training-the-gnn)
- [Usage](#usage)
- [License](#license)

## Overview

This project is designed for in silico evolution of proteins towards structure and function prediction. It leverages ESM-2 protein language models to generate embeddings for protein sequences, constructs graphs representing PPI networks, and applies a GraphSAGE-style GNN for edge (interaction) prediction and edge weight estimation.

## Features

- **Flexible protein encoding** using ESM-2 models (15B, 3B, 650M).
- **Graph construction** using protein embeddings and custom edge input.
- **Extendable GNN architecture** focused on edge prediction and weight regression.
- **Customizable training pipeline** with negative edge sampling, batching, and split configuration.
- **PyTorch and PyTorch Geometric** powered.

## Highlights
- GNN training pipeline (GraphSAGE / Pool_SAGEConv variants).
- Edge sampling with stratified sampling and unsampled tracking (EdgeSampler).
- Curriculum learning support to enable training with and without graph-based predictions.
- Optuna-based HPO orchestration with JournalStorage for distributed runs.
- Surrogate-based HPO helpers (RandomForest) to leverage prior results.
- Final inference model is under development and will be published to a public model repo (e.g., Hugging Face).

---

## Installation
### **Installation Guide for PiPPINN**

This guide provides step-by-step instructions to set up the environment and install all necessary dependencies for the **PiPPINN** project.

---

### **1. Prerequisites**
Before proceeding, ensure the following are installed on your system:
- **Python**: Version 3.8 or higher (tested with Python 3.9).
- **CUDA**: If you plan to use a GPU, ensure CUDA is installed and properly configured. Check your CUDA version with:
  ```bash
  nvcc --version
  ```
- **pip**: Ensure you have the latest version of `pip`:
  ```bash
  python -m pip install --upgrade pip
  ```

---

### **2. Clone the Repository**
Clone the repository to your local machine:
```bash
git clone https://github.com/your-repo/PiPPINN.git
cd PiPPINN
```

---

### **3. Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

---

### **4. Install Dependencies**
Install the required Python packages using `pip`.

#### **4.1 Install PyTorch**
Install PyTorch with the appropriate CUDA version. Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the latest instructions.

For example:
- **For CUDA 12.4**:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **For CPU-only**:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

#### **4.2 Install PyTorch Geometric and Dependencies**
Install PyTorch Geometric and its dependencies. Use the appropriate command based on your PyTorch and CUDA versions.

For example, if using PyTorch 2.5.1 with CUDA 12.4:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

#### **4.3 Install Additional Dependencies**
Install the remaining dependencies listed in the repository:
```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not present, create one with the following content based on the imports in the repository:

```text
argparse
fair-esm
torch
torch-geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
pyg-lib
optuna
```

---

### **5. Verify Installation**
Run the following commands to verify that all dependencies are installed correctly:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch_geometric; print('PyTorch Geometric version:', torch_geometric.__version__)"
python -c "import torch_sparse; print('torch-sparse installed successfully!')"
```

---

### **6. Run the Project**
Once all dependencies are installed, you can run the project scripts. For example:
```bash
python OptimizeHyperparams.py --input <path-to-data> --output <output-path>
python Retrain.py --input_data <path-to-data> --parameters <path-to-best-params>
```

---

### **7. Troubleshooting**
- **Missing Dependencies**:
  If you encounter missing dependencies, ensure you have installed all required libraries using the commands above.
- **Version Mismatch**:
  Ensure that the versions of PyTorch, CUDA, and PyTorch Geometric are compatible. Refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for compatibility details.
- **CUDA Issues**:
  If CUDA is not detected, ensure that your GPU drivers and CUDA toolkit are properly installed.


---

### **9. Deactivate the Virtual Environment**
When you're done, deactivate the virtual environment:
```bash
deactivate
```

---

### **Conclusion**
You are now ready to use the **PiPPINN** project. If you encounter any issues, feel free to reach out or consult the documentation.

## Pipeline

### Key files
- `TrainUtils.py` — core models, EdgeSampler, Decoders, training and batch utilities.
- `OptimizeHyperparams.py` — Optuna study controller, objective wrapper, surrogate support.
  - Create study with sampler and pruner:
    ```py
    sampler = TPESampler(seed=SEED, multivariate=True)
    pruner = HyperbandPruner(...)
    study = optuna.create_study(study_name=..., direction="minimize", sampler=sampler, pruner=pruner, storage=JournalStorage(JournalFileBackend(journal_file)), load_if_exists=True)
    ```
  - For distributed runs, ensure every worker uses the same `journal_file`, `study_name`, and search-space code.
  - To resume: pass `load_if_exists=True` and run `study.optimize(objective, n_trials=ADDITIONAL_TRIALS)`.
- `Retrain.py` — small retrain helper.
- `Evaluate_Trained.py` — batch loader / evaluation.

---

### 1. Protein Encoding

Encode protein sequences into embeddings using the ESM-2 model.

**Script:** `EncodeProteins.py`

**Usage:**
```bash
python EncodeProteins.py --input <sequence_file.tsv> --modelname 3B --outfile <output_embeddings.pt>
```
- `--input/-i`: Tab-separated file (ID, sequence).
- `--modelname/-m`: ESM-2 model [15B, 3B, 650M].
- `--outfile/-o`: Output file for embeddings.

### 2. Graph Preparation

Build a PyTorch Geometric graph from embeddings and an edge list.

**Script:** `Generate_PyG_Graph_Data.py`

**Usage:**
```bash
python Generate_PyG_Graph_Data.py --node-embeddings <output_embeddings.pt> --edges <edge_list.tsv> --output <graph.pt>
```
- `--node-embeddings/-n`: Output of previous step.
- `--edges/-e`: Tab-separated edge file (node1, node2 [,weight]).
- `--output/-o`: Output PyG graph file.

### 3. Training the GNN

Train a GraphSAGE-based GNN on your constructed graph.

#### 3.21 Hyperparameter optimization (HPO)
- HPO controller: `OptimizeHyperparams.py`.
- Supports Optuna with `JournalStorage` + `JournalFileBackend` for distributed workers.
- Recommended setup for distributed Bayesian HPO:
```py
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

sampler = TPESampler(seed=SEED, multivariate=True)
pruner = HyperbandPruner(min_resource=15, reduction_factor=3)
storage = JournalStorage(JournalFileBackend("/shared/optuna_journal.log"))
study = optuna.create_study(study_name="PiPPINN_HPO", direction="minimize",
                            sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=NUM_TRIALS)
```

  #### 3.1 Train / Retrain
- For fine-tuning or retraining a saved model, use `Retrain.py`:
```bash
python Retrain.py --input_data <graph.pt> --parameters <best_params.json> --output <retrained_model.pt>
```

### Perform predictions

## Usage

See the above pipeline for step-by-step commands. For each script, use the `--help` flag for detailed options and descriptions.

## License

This repository is licensed under the terms of the [LICENSE](LICENSE) file.
