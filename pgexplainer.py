import numpy as np
from typing import Optional
from math import sqrt
from torch import Tensor
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


EPS = 1e-6


class PGExplainer(nn.Module):

    def __init__(self, model, in_channels: int, epochs: int = 20,
                 lr: float = 0.005, coff_size: float = 0.01, coff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 1.0, num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        self.model = model
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.num_hops = num_hops
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))

    def __set_masks__(self, data, edge_mask: Tensor = None):
        (N, F), E = data.x.size(), data.edge_index.size(1)
        init_bias = self.init_bias
        std = nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __loss__(self, prob: Tensor, ori_pred: int):
        if len(prob.shape) == 1 and prob.shape[0] == 1:
            ori_pred = torch.tensor(ori_pred, dtype=torch.float)
            pred_loss = F.binary_cross_entropy(prob, ori_pred)
        else:
            logit = prob[ori_pred]
            logit = logit + EPS
            pred_loss = - torch.log(logit)
        # return pred_loss
        # size
        edge_mask = self.mask_sigmoid
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def run(self, data, embed, tmp: float = 1.0, training: bool = False):
        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        col, row = data.edge_index
        f1 = embed[col]
        f2 = embed[row]
        f12self = torch.cat([f1, f2], dim=-1)

        # edge weight
        h = f12self
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        mask_sparse = torch.sparse_coo_tensor(
            data.edge_index, values, (nodesize, nodesize)
        )
        self.mask_sigmoid = mask_sparse.to_dense()
        edge_mask = self.mask_sigmoid[data.edge_index[0], data.edge_index[1]]

        self.__clear_masks__()
        self.__set_masks__(data, edge_mask)

        out = self.model(data).view(-1)
        out = nn.Softmax(dim=-1)(out)

        self.__clear_masks__()
        return out, edge_mask

    def train_explanation_network(self, loader, label=0):
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)
        with torch.no_grad():
            self.model.eval()
            emb_dict = {}
            ori_pred_dict = {}
            for i, data in enumerate(tqdm(loader.dataset)):
                data.__setattr__('batch', torch.zeros(data.num_nodes, dtype=torch.long))
                out = self.model(data).view(-1)
                emb = self.model.get_emb(data)
                emb_dict[i] = emb.data
                if len(out.shape) == 1 and out.shape[0] == 1:
                    ori_pred_dict[i] = torch.round(out).data
                else:
                    ori_pred_dict[i] = out.argmax(dim=-1).data

        # train mask generator
        for epoch in range(self.epochs):
            loss = 0.0
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            optimizer.zero_grad()
            for i, data in enumerate(tqdm(loader.dataset)):
                data.__setattr__('batch', torch.zeros(data.num_nodes, dtype=torch.long))
                if ori_pred_dict[i] != label:
                    continue
                out, _ = self.run(data, embed=emb_dict[i], tmp=tmp, training=True)
                loss_tmp = self.__loss__(out, ori_pred_dict[i])
                loss_tmp.backward()
                loss += loss_tmp.item()
            optimizer.step()
        self.elayers.eval()

    def explain(self, data):
        (N, F), E = data.x.size(), data.edge_index.size(1)

        self.__clear_masks__()
        embed = self.model.get_emb(data)

        _, edge_mask = self.run(data, embed, tmp=1.0, training=False)
        node_mask = torch.zeros(N)
        for i, mask in enumerate(edge_mask):
            u, v = data.edge_index[0][i], data.edge_index[1][i]
            if edge_mask[i] > node_mask[u]:
                node_mask[u] = edge_mask[i]
            if edge_mask[i] > node_mask[v]:
                node_mask[v] = edge_mask[i]

        return node_mask, edge_mask
