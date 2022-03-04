# Illuminati

This repository contains the source code for the paper, `Illuminati: Towards Explaining Graph Neural Networks for Cybersecurity Analysis` by Haoyu He, Yuede Ji and H. Howie Huang ([EuroS&P 2022](https://www.ieee-security.org/TC/EuroSP2022/)).

We provide the implementation of public datasets.

## Requirements

We use [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#) for implementation. Please follow the instruction to install Pytorch Geometric.

## Datasets

The datasets can be loaded from Pytorch Geometric. The datasets we use in the paper are:

\["BBBP", "Mutagenicity", "BA-2motifs"\] for graph classification. <br/>
\["Cora", "Citeseer"\] for node classification.

We modify the dataset loader from [DIG](https://github.com/divelab/DIG).

## Using Illuminati

`TASK` as `graph` for graph classification, `node` for node classification.

### Configuring the project

We use `configs.json` to control this project. Specifically,

\[mode\]: the choice of explanation methods, {0: GNNExplainer or Illuminati, 1: PGM-Explainer, 2: PGExplainer} <br/>
\[node\]: whether to estimate node importance scores, i.e., GNNExplainer or Illuminati <br/>
\[synchronize\]: synchronized attribute mask learning <br/>
\[agg1 & agg2\]: aggregation functions for node importance scores \[mean, min, max, sum\] <br/>
\[sample\]: the number of samples for PGM-Explainer

### Training a GCN model

```
python train_TASK.py
```

## Cite

```
```
