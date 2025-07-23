import torch
import torch.nn as nn
import torch.nn.functional as F 

from models.complex_act import ComReLU
from operators.graph_operator.directed.magnetic_adaptive_operator import MagAdaptiveGraphOp


class MagneticAdaptiveplusGraphConvolution(nn.Module):
    def __init__(self, prop_steps, num_layers, use_att, feat_dim, hidden_dim, output_dim, dropout, task_level):
        super(MagneticAdaptiveplusGraphConvolution, self).__init__()
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

        self.prop_steps = prop_steps
        self.num_layers = num_layers
        self.comrelu = ComReLU()
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Linear(2, 1)
        
        self.use_att = False if use_att == 0 else True
        if self.use_att:
            self.real_att = nn.Linear(feat_dim, 1)
            self.imag_att = nn.Linear(feat_dim, 1)
        
        self.real_linear = nn.ModuleList()
        self.imag_linear = nn.ModuleList()
        self.real_linear.append(nn.Linear(feat_dim, hidden_dim))
        self.imag_linear.append(nn.Linear(feat_dim, hidden_dim))
        for layer in range(num_layers-1):
            self.real_linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.imag_linear.append(nn.Linear(hidden_dim, hidden_dim))
        
        if task_level == "node":
            self.real_imag_linear = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.real_imag_linear = nn.Linear(hidden_dim*4, output_dim)
        
    def forward(self, real_feature, imag_feature, device):
        # find for every edge a q-value
        structure_info = torch.stack((self.edge_entropy, self.edge_cluster_coefficient), dim=1)
        structure_encoding = self.encoding(structure_info).reshape(-1)

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
        
        real_feature_list = torch.unsqueeze(real_feature, dim=0)
        imag_feature_list = torch.unsqueeze(imag_feature, dim=0)
        
        for step in range(self.prop_steps):
            real_real_x_prop = torch.sparse.mm(real_adj, real_feature_list[-1])
            imag_imag_x_prop = torch.sparse.mm(imag_adj, imag_feature_list[-1])
            imag_real_x_prop = torch.sparse.mm(imag_adj, real_feature_list[-1])
            real_imag_x_prop = torch.sparse.mm(real_adj, imag_feature_list[-1])

            layer_out_real = real_real_x_prop - imag_imag_x_prop
            layer_out_imag = imag_real_x_prop + real_imag_x_prop
            
            layer_out_real = torch.unsqueeze(layer_out_real, dim=0)
            layer_out_imag = torch.unsqueeze(layer_out_imag, dim=0)
            
            real_feature_list = torch.cat((real_feature_list, layer_out_real), dim=0)
            imag_feature_list = torch.cat((imag_feature_list, layer_out_imag), dim=0)

        if self.use_att:
            real_weight_ = self.real_att(real_feature_list).permute(1, 0, 2)
            imag_weight_ = self.imag_att(imag_feature_list).permute(1, 0, 2)
            
            real_weight_list = F.softmax(torch.sigmoid(real_weight_), dim=1)
            imag_weight_list = F.softmax(torch.sigmoid(imag_weight_), dim=1)
            
            real_feature_reshape = real_feature_list.permute(1, 2, 0)
            real_weighted_feat = torch.bmm(real_feature_reshape, real_weight_list).squeeze(dim=2)
            
            imag_feature_reshape = imag_feature_list.permute(1, 2, 0)
            imag_weighted_feat = torch.bmm(imag_feature_reshape, imag_weight_list).squeeze(dim=2)
        else:
            real_weighted_feat, imag_weighted_feat = real_feature_list[-1], imag_feature_list[-1]

        real_x, imag_x = real_weighted_feat, imag_weighted_feat
        
        for layer in range(self.num_layers):
            layer2_out_real = self.real_linear[layer](real_x)
            layer2_out_imag = self.imag_linear[layer](imag_x)
            
            real_x, imag_x = self.comrelu(layer2_out_real, layer2_out_imag)
            real_x, imag_x = self.dropout(real_x), self.dropout(imag_x)

        if self.query_edges is None:
            x = torch.cat((real_x, imag_x), dim=-1)
        else:
            x = torch.cat((real_x[self.query_edges[:, 0]], real_x[self.query_edges[:, 1]], 
                           imag_x[self.query_edges[:, 0]], imag_x[self.query_edges[:, 1]]), dim=-1)
            
        x = self.real_imag_linear(x)
        return x


class MAPplus(nn.Module):
    def __init__(self, prop_steps, num_layers, q, use_att, feat_dim, hidden_dim, output_dim, dropout, task_level, label=None, test_idx=None):
        super(MAPplus, self).__init__()
        self.naive_graph_op = MagAdaptiveGraphOp(q)
        self.base_model = MagneticAdaptiveplusGraphConvolution(prop_steps, num_layers, use_att, \
                                                                    feat_dim, hidden_dim, output_dim, dropout, task_level)

        self.post_graph_op = None

        # self.label = label
        # self.test_idx = test_idx

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

        output = self.base_model(real_processed_feature, imag_processed_feature, device)

        # if torch.equal(self.test_idx, idx):
        #     self.base_model.soft_label = F.one_hot(self.label).float()
        #     output_detach = output.detach().cpu()
        #     self.base_model.soft_label[idx] = torch.softmax(output_detach[idx], dim=1)

        return output[idx] if self.base_model.query_edges is None else output
