import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models import GCN_

with open("configs.json") as config_file:
    configs = json.load(config_file)
    dataset_name = configs.get("dataset_name").get("node")

model_file = './src/' + dataset_name + '.pt'
dataset = Planetoid(root='./datasets', name=dataset_name, split='public')
data = dataset.data
feat_norm = NormalizeFeatures()
data = feat_norm(data)

gnn = GCN_(in_channels=dataset.num_node_features, hidden_channels=64, num_layers=2,
           out_channels=dataset.num_classes, jk='last')
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
gnn.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = gnn(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
gnn.eval()
pred = gnn(data.x, data.edge_index)
pred = pred.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

gnn.save(model_file)
