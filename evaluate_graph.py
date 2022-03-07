import csv
import json

import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.load_datasets import get_dataset, get_dataloader
import torch_geometric.transforms as T
from models import GCN, softmax_accuracy
from explainer import subgraph_by_node

with open("configs.json") as config_file:
    configs = json.load(config_file)
    explainer_args = configs.get("explainer")
    dataset_name = configs.get("dataset_name").get("graph")

epochs = 5000
loop = True
pooling = {'mutagenicity': ['max', 'mean', 'sum'],
           'ba_2motifs': ['max'],
           'bbbp': ['max', 'mean', 'sum']}
if dataset_name == 'ba_2motifs':
    early_stop = 500
    loop = False
dataset = get_dataset(dataset_dir="./datasets", dataset_name=dataset_name)
dataset.data.x = dataset.data.x.float()
normalize = T.NormalizeFeatures()
dataset.data = normalize(dataset.data)
dataset.data.y = dataset.data.y.squeeze().long()
mode = explainer_args.get("mode")
node = bool(explainer_args.get("node"))
top_node = explainer_args.get("node_rate")
data_loader = get_dataloader(dataset, batch_size=1, random_split_flag=True,
                             data_split_ratio=[0.8, 0.1, 0.1], seed=2)

model = GCN(n_feat=dataset.num_node_features,
            n_hidden=20,
            n_class=dataset.num_classes,
            pooling=pooling[dataset_name],
            loop=loop)
model_file = './src/' + dataset_name + '.pt'
model.load(model_file)
model.eval()

accuracy_test = []
for i, data in enumerate(data_loader['test']):
    logit = model(data)
    acc = softmax_accuracy(logit, data.y.float())
    accuracy_test.append(acc.item())
print(sum(accuracy_test), sum(accuracy_test)/len(accuracy_test))
print(len(dataset), dataset.data.x.shape[0]/len(dataset))

results_path = "./node_masks/" + dataset_name + "/"
print(dataset, "mode: ", mode, "node: ", node)

acc_p = 0
acc_n = 0
results_p = []
results_n = []
for i, data in enumerate(tqdm(data_loader['test'])):
    num_nodes = torch.max(data.edge_index) + 1
    if top_node >= num_nodes:
        continue
    node_mask = []
    if mode == 0:
        file_path = results_path + str(i) + "_" + str(node) + ".csv"
    elif mode == 1:
        file_path = results_path + str(i) + "_" + "pgm" + ".csv"
    else:
        file_path = results_path + str(i) + "_" + "pg" + ".csv"
    with open(file_path, newline='') as filehandle:
        cw = csv.reader(filehandle)
        for listitem in cw:
            node_mask.append(float(listitem[0]))
    node_mask = torch.tensor(node_mask, dtype=torch.float)
    logit = model(data)
    logp = nn.Softmax(dim=-1)(logit).view(-1)
    pred_label = torch.argmax(logp)
    data_p = subgraph_by_node(data, 1 - node_mask, num_nodes - top_node)
    data_p.__setattr__('batch', torch.zeros(data_p.num_nodes, dtype=torch.long))
    data_n = subgraph_by_node(data, node_mask, top_node)
    data_n.__setattr__('batch', torch.zeros(data_n.num_nodes, dtype=torch.long))
    logit_p = model(data_p)
    logp_p = nn.Softmax(dim=-1)(logit_p).view(-1)
    prob_p = logp[pred_label] - logp_p[pred_label]
    results_p.append(prob_p.item())
    if logp_p[pred_label] >= 0.5:
        acc_p += 1
    logit_n = model(data_n)
    logp_n = nn.Softmax(dim=-1)(logit_n).view(-1)
    prob_n = logp[pred_label] - logp_n[pred_label]
    results_n.append(prob_n.item())
    if logp_n[pred_label] >= 0.5:
        acc_n += 1

print("EP-")
print(acc_p, acc_p / len(results_p), 1 - acc_p / len(results_p))
print(sum(results_p), sum(results_p) / len(results_p))
print("EP+")
print(acc_n, acc_n / len(results_n), 1 - acc_n / len(results_n))
print(sum(results_n), sum(results_n) / len(results_n))