# PPI-GNN

PPI-GNN is a framework for protein-protein interaction (PPI) prediction using Graph Neural Networks (GNNs). This repository provides a pipeline to encode protein sequences, construct graphs suitable for PyTorch Geometric, and train a custom GNN model to predict interactions.

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

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/BharatRaviIyengar/PPI-GNN.git
cd PPI-GNN

# Install Python dependencies (modify as needed)
pip install torch torch-geometric esm torch-scatter
```

## Pipeline

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

**Script:** `TrainGNN.py`

**Usage:**
```bash
python TrainGNN.py --input <graph.pt> --epochs 50 --output <trained_model.pt>
```
- `--input/-i`: PyG graph file.
- `--epochs/-e`: Number of training epochs.
- `--output/-o`: Output model file.

## Usage

See the above pipeline for step-by-step commands. For each script, use the `--help` flag for detailed options and descriptions.

## License

This repository is licensed under the terms of the [LICENSE](LICENSE) file.
