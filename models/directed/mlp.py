import torch
import torch.nn as nn

from models.complex_act import ComReLU


class MLP(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()

        self.fc1_real = nn.Linear(feat_dim, hidden_dim)

        self.fc1_imag = nn.Linear(feat_dim, hidden_dim)

        self.post_graph_op = None

        self.dp1 = nn.Dropout()

        self.comrelu = ComReLU()

        self.fc2_real = nn.Linear(hidden_dim, hidden_dim)

        self.fc2_imag = nn.Linear(hidden_dim, hidden_dim)

        self.clf = nn.Linear(2 * hidden_dim, out_dim)

    def preprocess(self, adj, feature):

        self.real_processed_feature = torch.FloatTensor(feature)
        self.imag_processed_feature = torch.FloatTensor(feature)

    def model_forward(self, idx, device, ori=False):
        return self.forward(idx, device, ori)

    def forward(self, idx, device, ori):

        feat_real, feat_imag = self.real_processed_feature.to(device), self.imag_processed_feature.to(device)

        x_real_real = self.fc1_real(feat_real)
        x_imag_imag = self.fc1_imag(feat_imag)
        x_imag_real = self.fc1_imag(feat_real)
        x_real_imag = self.fc1_real(feat_imag)

        layer1_real_out = x_real_real - x_imag_imag
        layer1_imag_out = x_real_imag - x_imag_real

        layer1_real_out, layer1_imag_out = self.comrelu(layer1_real_out, layer1_imag_out)
        layer1_real_out, layer1_imag_out = self.dp1(layer1_real_out), self.dp1(layer1_imag_out)

        x_real_real = self.fc2_real(layer1_real_out)
        x_imag_imag = self.fc2_imag(layer1_imag_out)
        x_real_imag = self.fc2_real(layer1_imag_out)
        x_imag_real = self.fc2_imag(layer1_real_out)

        layer2_real_out = x_real_real - x_imag_imag
        layer2_imag_out = x_real_imag - x_imag_real

        layer2_real_out, layer2_imag_out = self.comrelu(layer2_real_out, layer2_imag_out)
        real_x, imag_x = self.dp1(layer2_real_out), self.dp1(layer2_imag_out)

        x = torch.cat((real_x, imag_x), dim=-1)

        output = self.clf(x)

        return output[idx]

