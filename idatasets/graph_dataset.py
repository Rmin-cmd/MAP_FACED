import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np


class GraphDataset(InMemoryDataset):
    def __init__(self, args, name, root, transform=None, pre_transform=None):
        self.args = args
        self.name = name
        self.root = root
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.

        # This is a placeholder for the actual data loading logic.
        # You should replace this with your own data loading code.
        num_graphs = 123
        num_nodes = 30
        num_features = 5

        data_list = []
        for i in range(num_graphs):
            edge_index = torch.randint(0, num_nodes, (2, 100), dtype=torch.long)
            x = torch.randn(num_nodes, num_features)
            y = torch.randint(0, 5, (1,), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    @property
    def num_features(self):
        return self.data.x.shape[1]

    @property
    def num_classes(self):
        return self.data.y.max().item() + 1

    @property
    def train_dataset(self):
        return self.data

    @property
    def val_dataset(self):
        return self.data

    @property
    def test_dataset(self):
        return self.data
