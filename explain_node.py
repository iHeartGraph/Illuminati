import csv
import json
import os
import torch

from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models import GCN_
from explainer import Explainer


with open("configs.json") as config_file:
    configs = json.load(config_file)
    explainer_args = configs.get("explainer")
    dataset_name = configs.get("dataset_name").get("node")

node = bool(explainer_args.get("node"))
model_file = './src/' + dataset_name + '.pt'
dataset = Planetoid(root='./datasets', name=dataset_name, split='public')
data = dataset.data
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

results_path = './node_masks/' + dataset_name + '/'
if not os.path.exists(results_path):
    os.makedirs(results_path)
test_idx = torch.nonzero(data.test_mask).view(-1)
explainer = Explainer(gnn, agg1=explainer_args.get("agg1"), agg2=explainer_args.get("agg2"), num_hops=2,
                      lr=explainer_args.get("lr"), epochs=explainer_args.get("epochs"))
for idx in tqdm(test_idx):
    node_feat_mask, edge_mask, node_mask, (node_idx, node_map) = \
        explainer.explain_node(data, loss_fc=None, idx=idx.item(), node=node, synchronize=explainer_args.get('synchronize'))
    node_idx_mask = torch.cat([node_idx.view(-1, 1).int(), node_mask.view(-1, 1)], dim=-1)
    file_path = results_path + str(idx.item()) + '_' + str(node) + '.csv'
    with open(file_path, 'w', newline='') as filehandle:
        cw = csv.writer(filehandle)
        cw.writerows(node_idx_mask.tolist())