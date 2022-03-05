import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ReLU
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, Size
from typing import Optional, Callable, List


from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing, GCNConv

torch.manual_seed(2020)


def softmax_accuracy(probs, all_labels):
    if len(probs.shape) == 1 and probs.shape[0] == all_labels.shape[0]:
        acc = (torch.round(probs) == all_labels).sum()
    else:
        acc = (torch.argmax(probs, dim=-1) == all_labels).sum()
    acc = torch.div(acc, len(all_labels) + 0.0)
    return acc


class GCNConv_(GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConv_, self).__init__(*args, **kwargs)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


class GCN(nn.Module):

    def __init__(self, n_feat, n_hidden, n_class, pooling, loop=True):
        super(GCN, self).__init__()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.conv1 = GCNConv_(in_channels=n_feat, out_channels=n_hidden, add_self_loops=loop, normalize=True)
        self.conv2 = GCNConv_(in_channels=n_hidden, out_channels=n_hidden, add_self_loops=loop, normalize=True)
        self.conv3 = GCNConv_(in_channels=n_hidden, out_channels=n_hidden, add_self_loops=loop, normalize=True)
        self.drop = nn.Dropout(p=0.2)
        self.pooling = pooling
        self.lin = nn.Linear(in_features=n_hidden * len(self.pooling), out_features=n_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.embeddings(x, edge_index)
        out = []
        if 'max' in self.pooling:
            out.append(gnn.global_max_pool(h, data.batch))
        if 'mean' in self.pooling:
            out.append(gnn.global_mean_pool(h, data.batch))
        if 'sum' in self.pooling:
            out.append(gnn.global_add_pool(h, data.batch))
        out = torch.cat(out, dim=-1)
        out = self.lin(out)

        return out

    def get_emb(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.embeddings(x, edge_index)
        return h

    def embeddings(self, x, edge_index):
        out1 = self.conv1(x, edge_index)
        out1 = F.normalize(out1, p=2, dim=1)
        out1 = self.activation1(out1)
        out2 = self.conv2(out1, edge_index)
        out2 = F.normalize(out2, p=2, dim=1)
        out2 = self.activation2(out2)
        out3 = self.conv3(out2, edge_index)
        out3 = F.normalize(out3, p=2, dim=1)
        out3 = self.activation3(out3)
        return out3

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class GCN_(BasicGNN):

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv_(in_channels, out_channels, **kwargs)

    def get_emb(self, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index
        h = self.embeddings(x, edge_index, *args, **kwargs)
        return h

    def embeddings(self, x: Tensor, edge_index: Adj, *args, **kwargs):
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))