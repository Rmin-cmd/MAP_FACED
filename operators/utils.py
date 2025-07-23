import numpy as np

import torch

from torch_scatter import scatter_add
from torch_sparse import coalesce


def adj_to_directed_symmetric_map_norm(adj, q):
        num_nodes = adj.shape[0]
        row, col = torch.tensor(adj.row, dtype=torch.long), torch.tensor(adj.col, dtype=torch.long)
        edge_weight = torch.tensor(adj.data)

        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
        theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
        edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

        edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, "add")
        edge_weight_sym = edge_attr[:, 0]
        edge_weight_sym = edge_weight_sym / 2
        loop_weight_sym = torch.ones((num_nodes))
        edge_weight_sym = torch.hstack((edge_weight_sym, loop_weight_sym))
        loop_edge_u_v = torch.linspace(0, num_nodes - 1, steps=num_nodes, dtype=int)
        loop_edge_index_u = torch.hstack((edge_index_sym[0], loop_edge_u_v))
        loop_edge_index_v = torch.hstack((edge_index_sym[1], loop_edge_u_v))
        edge_index_sym = torch.vstack((loop_edge_index_u, loop_edge_index_v))

        theta_weight = edge_attr[:, 1]
        loop_weight = torch.zeros((num_nodes))
        theta_weight = torch.hstack((theta_weight, loop_weight))

        row, col = edge_index_sym[0], edge_index_sym[1]
        deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt_left = torch.pow(deg, -0.5)
        deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
        deg_inv_sqrt_right = torch.pow(deg, -0.5)
        deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)

        exp_weight_q = 2 * np.pi * q * theta_weight

        edge_weight_sym = deg_inv_sqrt_left[row] * edge_weight_sym * deg_inv_sqrt_right[col]

        entropy = -deg / sum(deg) * torch.log(deg / sum(deg))
        edge_entropy = entropy[row] * entropy[col]

        triple_motif = torch.tensor(adj.dot(adj).multiply(adj.transpose()).sum(1)).reshape(-1)
        in_deg = torch.tensor(adj.sum(0)).reshape(-1)
        out_deg = torch.tensor(adj.sum(1)).reshape(-1)
        cluster_coefficient = triple_motif / (in_deg * out_deg)
        cluster_coefficient = torch.nan_to_num(cluster_coefficient, 0.0)
        edge_cluster_coefficient = cluster_coefficient[row] * cluster_coefficient[col]
        
        return row, col, edge_weight_sym, exp_weight_q, edge_entropy, edge_cluster_coefficient, num_nodes
