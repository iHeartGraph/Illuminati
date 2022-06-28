# Illuminati

This repository contains the source code for the paper, <b>"Illuminati: Towards Explaining Graph Neural Networks for Cybersecurity Analysis"</b> by Haoyu He, Yuede Ji and H. Howie Huang ([EuroS&P 2022](https://www.ieee-security.org/TC/EuroSP2022/)).

We provide the implementation of public datasets for the purpose of reproduction.

## Requirements

We use [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#) for implementation. Please follow the instruction to install Pytorch Geometric.

## Datasets

The datasets can be loaded from Pytorch Geometric. The datasets we use in the paper are:

`["BBBP", "Mutagenicity", "BA-2motifs"]` for graph classification. <br/>
`["Cora", "Citeseer"]` for node classification.

We modify the dataset loader from [DIG](https://github.com/divelab/DIG).

## Using Illuminati

`TASK` as `graph` for graph classification, `node` for node classification.

### Configuring the project

We use `configs.json` to control this project. Specifically,

```
mode - the choice of explanation methods {0: GNNExplainer or Illuminati, 1: PGM-Explainer, 2: PGExplainer} 
node - whether to estimate node importance scores, i.e., GNNExplainer or Illuminati 
synchronize - synchronized attribute mask learning 
agg1 & agg2 - aggregation functions for node importance scores [mean, min, max, etc.] 
sample - the number of samples for PGM-Explainer
```

### Training a GNN model

```
$ python train_TASK.py
```

We save the model with the best validation result while training. We also provide early stop.

### Explaining a GNN model

```
$ python explain_TASK.py
```

This is to explain the whole dataset. We save node importance scores for each graph as a single ".csv" file. The time complexity can be evaluated here.

### Evaluating an explanation method

```
$ python evaluate_TASK.py
```

Here, we provide Essentialness Percentage (EP) and probability reduction for explained subgraphs and remaining subgraphs.

## Cite

```
@INPROCEEDINGS{illuminati22,  
  author={He, Haoyu and Ji, Yuede and Huang, H. Howie},  
   booktitle={2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P)},   
   title={Illuminati: Towards Explaining Graph Neural Networks for Cybersecurity Analysis},   
   year={2022},  
   volume={},  
   number={},  
   pages={74-89},  
   doi={10.1109/EuroSP53844.2022.00013}}
```
