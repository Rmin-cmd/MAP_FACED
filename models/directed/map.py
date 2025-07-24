import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from models.complex_act import ComReLU
from operators.graph_operator.directed.magnetic_adaptive_operator import MagAdaptiveGraphOp


class Complex2LayerMAPGraphConvolution(nn.Module):
    def __init__(self, feat_dim, hidden_dim, output_dim, dropout, task_level):
        super(Complex2LayerMAPGraphConvolution, self).__init__()
        self.query_edges = None

        self.num_node = None

        self.row = None
        self.col = None
        self.indices = None
        self.edge_weight_sym = None
        self.exp_weight_q = None
        self.edge_entropy = None
        self.edge_cluster_coefficient = None
        self.soft_label = None
        self.task_level = task_level

        self.comrelu = ComReLU()
        self.dropout = nn.Dropout(dropout)
        self.real_imag_prop_fc1 = nn.Linear(feat_dim, hidden_dim)
        self.real_imag_prop_fc2 = nn.Linear(hidden_dim, hidden_dim)
        if task_level == "node":
            self.real_imag_linear = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.real_imag_linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, real_feature, imag_feature, batch=None):
        structure_encoding = self.edge_entropy + self.edge_cluster_coefficient

        if self.soft_label is None:
            edge_weight_q_real = torch.cos(self.exp_weight_q * structure_encoding)
            edge_weight_q_imag = torch.sin(self.exp_weight_q * structure_encoding)
        else:
            sim = torch.cosine_similarity(self.soft_label[self.row], self.soft_label[self.col], dim=1)
            feature_encoding = torch.acos(sim - 0.000001) / torch.pi * 2
            edge_weight_q_real = torch.cos(self.exp_weight_q * structure_encoding * feature_encoding)
            edge_weight_q_imag = torch.sin(self.exp_weight_q * structure_encoding * feature_encoding)

        edge_weight_real = self.edge_weight_sym * edge_weight_q_real
        edge_weight_imag = self.edge_weight_sym * edge_weight_q_imag

        real_adj = torch.sparse_coo_tensor(self.indices, edge_weight_real, [self.num_node, self.num_node])
        imag_adj = torch.sparse_coo_tensor(self.indices, edge_weight_imag, [self.num_node, self.num_node])

        real_real_x_prop, imag_imag_x_prop = torch.mm(real_adj, real_feature), torch.mm(imag_adj, imag_feature)
        imag_real_x_prop, real_imag_x_prop = torch.mm(imag_adj, real_feature), torch.mm(real_adj, imag_feature)

        real_real_x = self.real_imag_prop_fc1(real_real_x_prop)
        imag_imag_x = self.real_imag_prop_fc1(imag_imag_x_prop)
        imag_real_x = self.real_imag_prop_fc1(imag_real_x_prop)
        real_imag_x = self.real_imag_prop_fc1(real_imag_x_prop)

        layer1_out_real = real_real_x - imag_imag_x
        layer1_out_imag = imag_real_x + real_imag_x
        layer1_out_real, layer1_out_imag = self.comrelu(layer1_out_real, layer1_out_imag)
        layer1_out_real, layer1_out_imag = self.dropout(layer1_out_real), self.dropout(layer1_out_imag)

        real_real_x_prop, imag_imag_x_prop = torch.mm(real_adj, layer1_out_real), torch.mm(imag_adj, layer1_out_imag)
        imag_real_x_prop, real_imag_x_prop = torch.mm(imag_adj, layer1_out_real), torch.mm(real_adj, layer1_out_imag)

        real_real_x = self.real_imag_prop_fc2(real_real_x_prop)
        imag_imag_x = self.real_imag_prop_fc2(imag_imag_x_prop)
        imag_real_x = self.real_imag_prop_fc2(imag_real_x_prop)
        real_imag_x = self.real_imag_prop_fc2(real_imag_x_prop)

        layer2_out_real = real_real_x - imag_imag_x
        layer2_out_imag = imag_real_x + real_imag_x
        real_x, imag_x = self.comrelu(layer2_out_real, layer2_out_imag)
        real_x, imag_x = self.dropout(real_x), self.dropout(imag_x)

        if self.query_edges is None:
            x = torch.cat((real_x, imag_x), dim=-1)
        else:
            x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]],
                           imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)

        if self.task_level == "graph":
            x = global_add_pool(x, batch)

        x = self.real_imag_linear(x)
        return x


class MAP(nn.Module):
    def __init__(self, q, feat_dim, hidden_dim, output_dim, dropout, label, test_idx, task_level):
        super(MAP, self).__init__()
        self.naive_graph_op = MagAdaptiveGraphOp(q)
        self.base_model = Complex2LayerMAPGraphConvolution(feat_dim, hidden_dim, output_dim, dropout, task_level)

        self.post_graph_op = None

        self.label = label
        self.test_idx = test_idx
        self.task_level = task_level

        self.real_processed_feature = None
        self.imag_processed_feature = None

    def preprocess(self, data):
        if self.task_level == "node":
            self.base_model.row, self.base_model.col, self.base_model.edge_weight_sym, self.base_model.exp_weight_q, \
            self.base_model.edge_entropy, self.base_model.edge_cluster_coefficient, self.base_model.num_node \
                = self.naive_graph_op.construct_adj(data.adj)
            self.base_model.indices = torch.stack([self.base_model.row, self.base_model.col], dim=0)
            self.base_model.edge_entropy = self.normalize(self.base_model.edge_entropy)
            self.base_model.edge_cluster_coefficient = self.normalize(self.base_model.edge_cluster_coefficient)

            self.real_processed_feature = torch.FloatTensor(data.x)
            self.imag_processed_feature = torch.FloatTensor(data.x)
        else:
            self.base_model.num_node = data.num_nodes
            self.base_model.row, self.base_model.col, self.base_model.edge_weight_sym, self.base_model.exp_weight_q, \
            self.base_model.edge_entropy, self.base_model.edge_cluster_coefficient, _ \
                = self.naive_graph_op.construct_adj(data.edge_index, data.num_nodes)
            self.base_model.indices = torch.stack([self.base_model.row, self.base_model.col], dim=0)
            self.base_model.edge_entropy = self.normalize(self.base_model.edge_entropy)
            self.base_model.edge_cluster_coefficient = self.normalize(self.base_model.edge_cluster_coefficient)

    def normalize(self, x):
        if sum(x) != 0:
            x = x * len(x) / sum(x)
            x = (1 - torch.exp(-2 * x)) / (1 + torch.exp(-2 * x))
        return x

    def postprocess(self, adj, output):
        return output

    def model_forward(self, data, device, ori=None):
        return self.forward(data, device, ori)

    def forward(self, data, device, ori):
        if self.task_level == "node":
            self.base_model.row = self.base_model.row.to(device)
            self.base_model.col = self.base_model.col.to(device)
            self.base_model.indices = self.base_model.indices.to(device)
            self.base_model.edge_weight_sym = self.base_model.edge_weight_sym.to(device)
            self.base_model.exp_weight_q = self.base_model.exp_weight_q.to(device)
            self.base_model.edge_entropy = self.base_model.edge_entropy.to(device)
            self.base_model.edge_cluster_coefficient = self.base_model.edge_cluster_coefficient.to(device)
            real_processed_feature = self.real_processed_feature.to(device)
            imag_processed_feature = self.imag_processed_feature.to(device)
            output = self.base_model(real_processed_feature, imag_processed_feature)
            return output[data]
        else:
            self.preprocess(data)
            self.base_model.row = self.base_model.row.to(device)
            self.base_model.col = self.base_model.col.to(device)
            self.base_model.indices = self.base_model.indices.to(device)
            self.base_model.edge_weight_sym = self.base_model.edge_weight_sym.to(device)
            self.base_model.exp_weight_q = self.base_model.exp_weight_q.to(device)
            self.base_model.edge_entropy = self.base_model.edge_entropy.to(device)
            self.base_model.edge_cluster_coefficient = self.base_model.edge_cluster_coefficient.to(device)
            real_processed_feature = data.x.to(device)
            imag_processed_feature = data.x.to(device)
            output = self.base_model(real_processed_feature, imag_processed_feature, data.batch)
            return output
