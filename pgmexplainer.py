import math
import pandas as pd
from scipy import stats
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch


class PGMExplainer:
    def __init__(self,
                 model,
                 num_hops: Optional[int] = None,
                 perturb_x_list=None,
                 perturb_mode="mean",
                 perturb_indicator="diff"):
        self.model = model
        self.model.eval()
        self.num_hops = num_hops
        self.perturb_x_list = perturb_x_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator

    def perturb_features_on_node(self, data, data_ori, node_idx, random=0):
        (N, F), E = data.x.size(), data.edge_index.size(1)
        num_nodes = torch.max(data.edge_index) + 1

        x_perturb = data.x.clone()
        perturb_array = x_perturb[node_idx].clone()
        epsilon = 0.05 * torch.max(data.x, dim=0)[0]
        seed = torch.randint(2, (1,))[0]

        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if i in self.perturb_x_list:
                        if self.perturb_mode == "mean":
                            perturb_array[i] = torch.mean(data.x[:num_nodes, i])
                        elif self.perturb_mode == "zero":
                            perturb_array[i] = 0
                        elif self.perturb_mode == "max":
                            perturb_array[i] = torch.max(data.x[:num_nodes, i])
                        elif self.perturb_mode == "uniform":
                            perturb_array[i] = perturb_array[i] + torch.rand() * epsilon[i] * 2 - epsilon[i]
                            if perturb_array[i] < 0:
                                perturb_array[i] = 0
                            elif perturb_array[i] > torch.max(data_ori.x, dim=0)[0][i]:
                                perturb_array[i] = torch.max(data_ori.x, dim=0)[0][i]
        x_perturb[node_idx] = perturb_array
        return Data(x=x_perturb, edge_index=data.edge_index, y=data.y)

    def batch_peturb_x_on_nodes(self, data, num_samples, index_to_perturb,
                                percentage, p_threshold, pred_threshold):
        (N, F), E = data.x.size(), data.edge_index.size(1)
        num_nodes = torch.max(data.edge_index) + 1

        pred = self.model(data)
        if len(pred.shape) == 1 and pred.shape[0] == 1:
            soft_pred = torch.tensor([1 - pred, pred])
        else:
            soft_pred = nn.Softmax(dim=-1)(pred).view(-1)
        pred_label = torch.argmax(soft_pred).long()
        Samples = []
        for iteration in range(num_samples):
            data_perturb = Data(x=data.x, edge_index=data.edge_index, y=data.y)
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = torch.randint(100, (1,))[0]
                    if seed < percentage:
                        latent = 1
                        data_perturb = self.perturb_features_on_node(data_perturb, data, node, random=latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            data_perturb = Batch.from_data_list([data_perturb])
            pred_perturb = self.model(data_perturb)
            if len(pred_perturb.shape) == 1 and pred_perturb.shape[0] == 1:
                soft_pred_perturb = torch.tensor([1 - pred_perturb, pred_perturb])
            else:
                soft_pred_perturb = nn.Softmax(dim=-1)(pred_perturb).view(-1)
            pred_change = torch.max(soft_pred) - soft_pred_perturb[pred_label]
            sample.append(pred_change)
            Samples.append(sample)

        Samples = torch.tensor(Samples, dtype=torch.float)
        if self.perturb_indicator == "abs":
            Samples = torch.abs(Samples)

        top = int(num_samples / 4)
        top_idx = torch.argsort(Samples[:, num_nodes])[-top:]
        for i in range(num_samples):
            Samples[i, num_nodes] = 1 if i in top_idx else 0
        return Samples

    def explain_graph(self, data, num_samples=10, percentage=50,
                      top_node=None, p_threshold=0.05, pred_threshold=0.1):
        (N, F), E = data.x.size(), data.edge_index.size(1)
        num_nodes = torch.max(data.edge_index) + 1

        if top_node is None:
            top_node = int(num_nodes / 5)
        if top_node < 1.:
            top_node = torch.ceil(num_nodes * top_node).int()
        if top_node >= num_nodes:
            top_node = num_nodes

        '''round 1'''
        Samples = self.batch_peturb_x_on_nodes(data, int(num_samples / 2), range(num_nodes), percentage,
                                               p_threshold, pred_threshold)
        dt = pd.DataFrame(Samples.numpy())
        p_values = []
        target = num_nodes.item()
        for node in range(num_nodes):
            chi2, p = self.chi_square(node, target, [], dt)
            p_values.append(p)
        number_candidates = int(top_node * 2) if int(top_node * 2) < num_nodes else num_nodes
        p_values = torch.tensor(p_values, dtype=torch.float)
        p_values[p_values == 1.] = 0.
        _, candidate_nodes = torch.topk(p_values, number_candidates)

        '''round 2'''
        Samples = self.batch_peturb_x_on_nodes(data, num_samples, candidate_nodes, percentage,
                                               p_threshold, pred_threshold)
        dt = pd.DataFrame(Samples.numpy())
        p_values = []
        dependent_nodes = []
        target = num_nodes.item()
        for node in range(num_nodes):
            chi2, p = self.chi_square(node, target, [], dt)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)
        p_values = torch.tensor(p_values, dtype=torch.float)
        p_values[p_values == 1.] = 0.
        _, ind_top_p = torch.topk(p_values, top_node)
        pgm_nodes = ind_top_p
        return pgm_nodes, p_values, candidate_nodes

    @staticmethod
    def chi_square(X, Y, Z, data):
        """
        Modification of Chi-square conditional independence test from pgmpy
        Tests the null hypothesis that X is independent from Y given Zs.
        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the separating set that (potentially) makes X and Y independent.
            Default: []
        Returns
        -------
        chi2: float
            The chi2 test statistic.
        p_value: float
            The p_value, i.e. the probability of observing the computed chi2
            statistic (or an even higher value), given the null hypothesis
            that X _|_ Y | Zs.
        sufficient_data: bool
            A flag that indicates if the sample size is considered sufficient.
            As in [4], require at least 5 samples per parameter (on average).
            That is, the size of the data set must be greater than
            `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
            (c() denotes the variable cardinality).
        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
        [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4
        """
        # X = str(int(X))
        # Y = str(int(Y))
        if isinstance(Z, (frozenset, list, set, tuple)):
            Z = list(Z)
        Z = [str(int(z)) for z in Z]

        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }

        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )

        XYZ_state_counts = pd.crosstab(
            index=data[X], columns=[data[Y]] + [data[z] for z in Z],
            rownames=[X], colnames=[Y] + Z
        )

        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=list(range(1, len(Z) + 1)))  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = np.zeros(XYZ_state_counts.shape)

        r_index = 0
        for X_val in XYZ_state_counts.index:
            X_val_array = []
            if Z:
                for Y_val in XYZ_state_counts.columns.levels[0]:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + list(temp.to_numpy())
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1
            else:
                for Y_val in XYZ_state_counts.columns:
                    temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                    X_val_array = X_val_array + [temp]
                XYZ_expected[r_index] = np.asarray(X_val_array)
                r_index = +1

        observed = XYZ_state_counts.to_numpy().reshape(1, -1)
        expected = XYZ_expected.reshape(1, -1)
        observed, expected = zip(*((o, e) for o, e in zip(observed[0], expected[0]) if not (e == 0 or math.isnan(e))))
        chi2, significance_level = stats.chisquare(observed, expected)

        return chi2, significance_level
