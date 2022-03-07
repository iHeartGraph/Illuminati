import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.transforms import NormalizeFeatures

from models import GCN_


with open("configs.json") as config_file:
    configs = json.load(config_file)
    explainer_args = configs.get("explainer")
    dataset_name = configs.get("dataset_name").get("node")

node = bool(explainer_args.get("node"))
top_node = explainer_args.get("node_rate")

model_file = './src/' + dataset_name + '.pt'
dataset = Planetoid(root='./datasets', name=dataset_name, split='public')
data = dataset[0]
feat_norm = NormalizeFeatures()
data = feat_norm(data)
gnn = GCN_(in_channels=dataset.num_node_features, hidden_channels=64, num_layers=2,
           out_channels=dataset.num_classes, jk='last', normalize=True)
gnn.load(model_file)
gnn.eval()

pred = gnn(data.x, data.edge_index)
pred = pred.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

result_path = './node_masks/' + dataset_name + '/'
test_idx = torch.nonzero(data.test_mask).view(-1)

acc = 0
results = []
for idx in tqdm(test_idx):
    node_idx, edge_idx, node_map, edge_map = k_hop_subgraph(idx.item(), 2, data.edge_index, relabel_nodes=True)
    num_nodes = node_idx.shape[0]
    if top_node >= num_nodes:
        continue
    idx_sub = node_map[0]
    logit = gnn(data.x[node_idx], edge_idx)
    logp = F.softmax(logit, -1)[idx_sub]
    pred_idx = torch.argmax(logp)

    file_path = result_path + str(idx.item()) + "_" + str(node) + ".csv"
    with open(file_path, newline='') as filehandle:
        cw = csv.reader(filehandle)
        node_idx_mask = np.array(list(cw)).astype(float)
        node_idx = node_idx_mask[:, 0].astype(int)
        node_map = np.where(node_idx == idx.item())[0][0]
        node_mask = node_idx_mask[:, 1]
    top_k = np.argpartition(node_mask, -top_node)[-top_node:]
    top_k = torch.tensor(top_k, dtype=torch.long)
    edge_idx_, _ = subgraph(top_k, edge_idx)

    logit_ = gnn(data.x[node_idx], edge_idx_)
    logp_ = F.softmax(logit_, -1)[idx_sub]

    if logp_[pred_idx] >= 0.5:
        acc += 1
    prob = logp[pred_idx] - logp_[pred_idx]
    results.append(prob.item())

print(acc, acc / len(results))
print(sum(results), sum(results) / len(results))
