from typing import Optional
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, k_hop_subgraph

EPS = 1e-15


def agg(mask):
    result = torch.tensor(1.)
    mask = mask.view(-1)
    weight = mask.shape[0]
    zeros = 0
    for m in mask:
        if m == 0.:
            zeros += 1
            continue
        result *= m
    weight = weight - zeros
    result = torch.pow(result, 1. / weight) if weight != 0 else torch.tensor(0.)
    if weight != 0:
        result = torch.pow(result, (weight + zeros) / weight)
    return result


class Explainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs: int = 50, lr: float = 0.01,
                 agg1="max", agg2="max", num_hops: Optional[int] = None):
        super(Explainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.drop = nn.Dropout(p=0.2)

        if agg1 == "mean":
            self.agg1 = torch.mean
        elif agg1 == "min":
            self.agg1 = torch.min
        elif agg1 == "max":
            self.agg1 = torch.max
        elif agg1 == "sum":
            self.agg1 = torch.sum
        else:
            self.agg1 = agg
        if agg2 == "mean":
            self.agg2 = torch.mean
        elif agg2 == "min":
            self.agg2 = torch.min
        elif agg2 == "max":
            self.agg2 = torch.max
        elif agg2 == "sum":
            self.agg2 = torch.sum
        else:
            self.agg2 = agg

    def __get_indices__(self, data):
        (N, F), E = data.x.size(), data.edge_index.size(1)

        self.out_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        self.in_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        for n in range(N):
            self.out_edge_mask[n] = (data.edge_index[0] == n) & (data.edge_index[1] != n)
            self.in_edge_mask[n] = (data.edge_index[1] == n) & (data.edge_index[0] != n)

        self.out_degree = torch.zeros(N, dtype=torch.int)
        self.in_degree = torch.zeros(N, dtype=torch.int)
        for n in range(N):
            in_num = torch.sum(self.in_edge_mask[n], dtype=torch.int)
            out_num = torch.sum(self.out_edge_mask[n], dtype=torch.int)
            self.in_degree[n] = in_num
            self.out_degree[n] = out_num

        self.self_loop_mask = torch.zeros(N, dtype=torch.long)
        for e in range(E):
            if data.edge_index[0, e] == data.edge_index[1, e]:
                self.self_loop_mask[data.edge_index[0, e]] = e
        if self.self_loop_mask.sum() == 0:
            self.self_loop_mask = None

    def __set_masks__(self, data, node: bool = True, synchronize: bool = True, edge_mask=None):
        (N, F), E = data.x.size(), data.edge_index.size(1)
        num_nodes = N

        if edge_mask is not None:
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            return

        std = 0.1
        node_feat_mask = torch.randn(1, F) * std if not node else torch.randn(N, F) * std
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        edge_mask = torch.randn(E) * std
        if not node and self.self_loop_mask is not None:
            edge_mask[self.self_loop_mask] = torch.ones(num_nodes)
        if node and synchronize:
            node_feat_mask = torch.mean(node_feat_mask, dim=-1, keepdim=True)
        self.node_feat_mask = torch.nn.Parameter(node_feat_mask)
        self.edge_mask = edge_mask

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
        self.out_degree = None
        self.in_degree = None
        self.out_edge_mask = None
        self.in_edge_mask = None
        self.self_loop_mask = None
        self.node_feat_mask = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __loss__(self, logits, pred_label, loss_fc=None):
        if loss_fc is None:
            loss_fc = F.cross_entropy
            pred_label = pred_label.long()
            if len(logits.shape) == 1 and logits.shape[0] == 1:
                loss_fc = F.binary_cross_entropy
                pred_label = pred_label.float()
        loss = loss_fc(logits, pred_label)
        # return loss

        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def __refine_mask__(self, mask, beta=1., training=True):
        if training:
            random_noise = torch.rand(mask.shape)
            random_noise = torch.log(random_noise) - torch.log(1 - random_noise)
            s = (random_noise + mask) / beta
            s = s.sigmoid()
            z = s * 1.5 - 0.25
            z = z.clamp(0, 1)
        else:
            z = (mask / beta).sigmoid()

        return z

    def explain_graph(self, data, loss_fc, node: bool = True, synchronize: bool = False):
        self.model.eval()
        self.__clear_masks__()

        (N, F), E = data.x.size(), data.edge_index.size(1)

        with torch.no_grad():
            out = self.model(data)
            if len(out.shape) == 1 and out.shape[0] == 1:
                pred_label = torch.round(out)
            else:
                pred_label = out.argmax(dim=-1)

        self.__get_indices__(data)
        self.__set_masks__(data, node=node, synchronize=synchronize)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            node_feat_mask = self.__refine_mask__(
                self.node_feat_mask, beta=(epoch + 1) / self.epochs) if node else self.node_feat_mask.sigmoid()
            h = data.x * node_feat_mask
            data_tmp = Data(x=h, edge_index=data.edge_index, batch=torch.zeros(N, dtype=torch.long))
            edge_mask = self.__refine_mask__(self.edge_mask,
                                             beta=(epoch + 1) / self.epochs) if node else self.edge_mask.sigmoid()
            self.__set_masks__(data_tmp, edge_mask=edge_mask)
            out = self.model(data_tmp)
            loss = self.__loss__(out, pred_label, loss_fc)
            loss.backward()
            optimizer.step()

        node_feat_mask = \
            self.__refine_mask__(self.node_feat_mask, training=False) if node else self.node_feat_mask.sigmoid()
        edge_mask = \
            self.__refine_mask__(self.edge_mask, training=False) if node else self.edge_mask.sigmoid()
        node_mask = torch.zeros(node_feat_mask.shape[0])

        if node:
            node_feat_msg = torch.sum(node_feat_mask * data.x, dim=-1).view(-1)
            x = data.x.clone()
            x[x > 0.] = 1.
            node_feat_mask = node_feat_mask * x
            for n in range(N):
                idx = torch.nonzero(x[n])
                node_feat_msg[n] = agg(node_feat_mask[n, idx])
            for n in range(N):
                if self.out_degree[n] > 0 or self.in_degree[n] > 0:
                    out_masks = torch.zeros(1)
                    if self.out_degree[n] > 0:
                        out_masks = edge_mask[self.out_edge_mask[n]]
                    node_mask_out = out_masks * node_feat_msg[n]
                    node_mask_out = self.agg1(node_mask_out)
                    in_masks = edge_mask[self.self_loop_mask[n]] if self.self_loop_mask is not None else torch.zeros(1)
                    node_mask_in = in_masks * node_feat_msg[n]
                    if self.in_degree[n] > 0:
                        in_nodes = data.edge_index[0, self.in_edge_mask[n]]
                        in_masks = edge_mask[self.in_edge_mask[n]]
                        if self.self_loop_mask is not None:
                            in_masks = torch.cat((in_masks.view(-1), edge_mask[self.self_loop_mask[n]].view(-1)))
                        node_mask_in = in_masks * node_feat_msg[in_nodes]
                    node_mask_in = self.agg1(node_mask_in)

                    node_mask[n] = self.agg2(torch.cat((node_mask_in.view(-1), node_mask_out.view(-1))))
        else:
            node_mask = torch.zeros(N)
            for n in range(N):
                out_max = torch.tensor(0, dtype=torch.float)
                in_max = torch.tensor(0, dtype=torch.float)
                if self.out_degree[n] > 0:
                    out_max = torch.max(edge_mask[self.out_edge_mask[n]])
                if self.in_degree[n] > 0:
                    in_max = torch.max(edge_mask[self.in_edge_mask[n]])
                node_mask[n] = torch.max(out_max, in_max)

        self.__clear_masks__()

        return node_feat_mask, edge_mask, node_mask

    def explain_node(self, data, loss_fc, idx: int = 0, node: bool = True, synchronize: bool = False):
        self.model.eval()
        self.__clear_masks__()

        node_idx, edge_idx, node_map, edge_map = k_hop_subgraph(idx, self.num_hops, data.edge_index, relabel_nodes=True)
        idx_sub = node_map[0]
        data = Data(data.x[node_idx], edge_idx, y=data.y[node_idx])
        (N, F), E = data.x.size(), data.edge_index.size(1)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)[idx_sub]
            if len(out.shape) == 1 and out.shape[0] == 1:
                pred_label = torch.round(out)
            else:
                pred_label = out.argmax(dim=-1)

        self.__get_indices__(data)
        self.__set_masks__(data, node=node, synchronize=synchronize)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            node_feat_mask = self.__refine_mask__(
                self.node_feat_mask, beta=(epoch + 1) / self.epochs) if node else self.node_feat_mask.sigmoid()
            h = data.x * node_feat_mask
            edge_mask = self.__refine_mask__(
                self.edge_mask, beta=(epoch + 1) / self.epochs) if node else self.edge_mask.sigmoid()
            self.__set_masks__(data, edge_mask=edge_mask)
            out = self.model(h, data.edge_index)[idx_sub].unsqueeze(0)
            loss = self.__loss__(out, pred_label.unsqueeze(0), loss_fc)
            loss.backward()
            optimizer.step()

        node_feat_mask = \
            self.__refine_mask__(self.node_feat_mask, training=False) if node else self.node_feat_mask.sigmoid()
        edge_mask = \
            self.__refine_mask__(self.edge_mask, training=False) if node else self.edge_mask.sigmoid()
        node_mask = torch.zeros(node_feat_mask.shape[0])

        if node:
            node_feat_msg = torch.sum(node_feat_mask * data.x, dim=-1).view(-1)
            x = data.x.clone()
            x[x > 0.] = 1.
            node_feat_mask = node_feat_mask * x
            for n in range(N):
                if self.out_degree[n] > 0 or self.in_degree[n] > 0:
                    out_masks = torch.zeros(1)
                    if self.out_degree[n] > 0:
                        out_masks = edge_mask[self.out_edge_mask[n]]
                    node_mask_out = out_masks * node_feat_msg[n]
                    node_mask_out = self.agg1(node_mask_out)
                    in_masks = edge_mask[self.self_loop_mask[n]] if self.self_loop_mask is not None else torch.zeros(1)
                    node_mask_in = in_masks * node_feat_msg[n]
                    if self.in_degree[n] > 0:
                        in_nodes = data.edge_index[0, self.in_edge_mask[n]]
                        in_masks = edge_mask[self.in_edge_mask[n]]
                        if self.self_loop_mask is not None:
                            in_masks = torch.cat((in_masks.view(-1), edge_mask[self.self_loop_mask[n]].view(-1)))
                        node_mask_in = in_masks * node_feat_msg[in_nodes]
                    node_mask_in = self.agg1(node_mask_in)

                    node_mask[n] = self.agg2(torch.cat((node_mask_in.view(-1), node_mask_out.view(-1))))
        else:
            node_mask = torch.zeros(N)
            for n in range(N):
                out_max = torch.tensor(0, dtype=torch.float)
                in_max = torch.tensor(0, dtype=torch.float)
                if self.out_degree[n] > 0:
                    out_max = torch.max(edge_mask[self.out_edge_mask[n]])
                if self.in_degree[n] > 0:
                    in_max = torch.max(edge_mask[self.in_edge_mask[n]])
                node_mask[n] = torch.max(out_max, in_max)

        self.__clear_masks__()

        return node_feat_mask, edge_mask, node_mask, (node_idx, node_map)


def subgraph_by_node(data, node_mask, node_rate):
    data = data.clone()
    node_mask = node_mask.clone()
    x, edge_index = data.x, data.edge_index
    (N, F), E = x.size(), edge_index.size(1)

    num_nodes = torch.max(edge_index) + 1
    node_mask = node_mask[:num_nodes]
    if node_rate < 1.:
        node_rate = torch.round(num_nodes.float() * node_rate).int()

    if node_rate >= num_nodes or node_rate == 0:
        return None

    node_idx = []
    for i in range(node_rate):
        idx = torch.argmax(node_mask).item()
        node_idx.append(idx)
        node_mask[idx] = -1

    x_ = torch.zeros_like(x)
    x_[node_idx, :] = x[node_idx, :].clone()
    edge_index_, _ = subgraph(node_idx, edge_index, relabel_nodes=False)

    return Data(x=x_, edge_index=edge_index_)
