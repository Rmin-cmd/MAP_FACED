import torch
import torch.nn as nn
import torch.nn.functional as F 

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

        self.comrelu = ComReLU()
        self.dropout = nn.Dropout(dropout)
        self.real_imag_prop_fc1 = nn.Linear(feat_dim, hidden_dim)
        self.real_imag_prop_fc2 = nn.Linear(hidden_dim, hidden_dim)
        if task_level == "node":
            self.real_imag_linear = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.real_imag_linear = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, real_feature, imag_feature):
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

        self.real_processed_feature = None
        self.imag_processed_feature = None

    def preprocess(self, adj, feature):
        self.base_model.row, self.base_model.col, self.base_model.edge_weight_sym, self.base_model.exp_weight_q, \
            self.base_model.edge_entropy, self.base_model.edge_cluster_coefficient, self.base_model.num_node \
                = self.naive_graph_op.construct_adj(adj)
        self.base_model.indices = torch.stack([self.base_model.row, self.base_model.col], dim=0)
        self.base_model.edge_entropy = self.normalize(self.base_model.edge_entropy)
        self.base_model.edge_cluster_coefficient = self.normalize(self.base_model.edge_cluster_coefficient)

        self.real_processed_feature = torch.FloatTensor(feature)
        self.imag_processed_feature = torch.FloatTensor(feature)

    def normalize(self, x):
        if sum(x) != 0:
            x = x * len(x) / sum(x)
            x = (1 - torch.exp(-2*x)) / (1 + torch.exp(-2*x))
        return x

    def postprocess(self, adj, output):
        return output

    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):
        self.base_model.row = self.base_model.row.to(device)
        self.base_model.col = self.base_model.col.to(device)
        self.base_model.indices = self.base_model.indices.to(device)
        self.base_model.edge_weight_sym = self.base_model.edge_weight_sym.to(device)
        self.base_model.exp_weight_q = self.base_model.exp_weight_q.to(device)
        self.base_model.edge_entropy = self.base_model.edge_entropy.to(device)
        self.base_model.edge_cluster_coefficient = self.base_model.edge_cluster_coefficient.to(device)
        real_processed_feature = self.real_processed_feature.to(device)
        imag_processed_feature = self.imag_processed_feature.to(device)

        if self.base_model.soft_label is not None:
            self.base_model.soft_label = self.base_model.soft_label.to(device)

        if ori is not None:
            self.base_model.query_edges = ori

        output = self.base_model(real_processed_feature, imag_processed_feature)

        if torch.equal(self.test_idx, idx):
            self.base_model.soft_label = F.one_hot(self.label).float()
            output_detach = output.detach().cpu()
            self.base_model.soft_label[idx] = torch.softmax(output_detach[idx], dim=1)

        return output[idx] if self.base_model.query_edges is None else output
