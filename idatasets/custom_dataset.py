import os
import torch
import scipy.io as sio
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from typing import List
from utils import load_srt_de  # adapt import to wherever your label loader is
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_sparse import coalesce
from idatasets.base_data import Graph


class MeanBandCollapse(object):
    def __call__(self, data: Data) -> Data:
        # data.edge_attr_all: Tensor [B, C, C]
        # mean over B → [C, C]
        A = data.edge_attr_all.mean(dim=1)
        spcoo = A.to_sparse().coalesce()
        data.edge_index = spcoo.indices()
        data.edge_attr = spcoo.values()
        del data.edge_attr_all
        return data


class CustomDataset(Dataset):
    """
    A simple Dataset that builds per-fold Data objects and returns (Data, label) tuples.
    No internal collate is used; external DataLoader should supply a custom collate_fn if needed.
    """

    def __init__(self, args, root, split='train', transform=None, pre_transform=None):
        self.args = args
        self.root = root
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform

        # Build the list of Data and labels once
        self.graphs, self.labels = self._build_graphs()

    def _build_graphs(self):
        args = self.args
        # load PDC
        if args.pdc_path.endswith('.h5'):
            with h5py.File(args.pdc_path, 'r') as f:
                A_pdc = np.array(f['connectivity'])
        else:
            A_pdc = sio.loadmat(args.pdc_path)['data']

        # load features
        feat_mat = os.path.join(args.feature_root_dir,
                                f'de_lds_fold{args.fold}.mat')
        feature_pdc = sio.loadmat(feat_mat)['de_lds']

        # reshape by classes
        num_windows = 11
        if args.num_classes == 9:
            label_type = 'cls9'
        elif args.num_classes == 2:
            feature_pdc = feature_pdc.reshape(
                feature_pdc.shape[0], -1, num_windows, feature_pdc.shape[2]
            )
            vid_sel = list(range(12)) + list(range(16, 28))
            feature_pdc = feature_pdc[:, vid_sel, :, :]
            A_pdc = A_pdc[:, :, vid_sel, :, :, :]
            feature_pdc = feature_pdc.reshape(
                feature_pdc.shape[0], -1, feature_pdc.shape[3]
            )
            label_type = 'cls2'
        else:
            raise ValueError("args.num_classes must be 2 or 9")

        # labels
        label_repeat = load_srt_de(feature_pdc, True, label_type, num_windows)
        # label_repeat: [n_subs, n_total_windows]

        # 5) Determine train / val splits
        all_indices = np.arange(args.n_subs)
        fold_size = args.n_subs // args.n_folds
        start = fold_size * args.fold
        end = fold_size * (args.fold + 1)
        if args.fold == args.n_folds - 1:
            val_idx = all_indices[start:]
        else:
            val_idx = all_indices[start:end]
        train_idx = np.setdiff1d(all_indices, val_idx)
        if self.split == 'train':
            sel_idx = train_idx
        else:  # 'val' or 'test'
            sel_idx = val_idx

        graphs = []
        labels = []
        for i in tqdm(sel_idx, desc=f"Building {self.split} fold {args.fold}"):
            # adjacency for subject i: assume shape [?, nodes, nodes]
            # Here we pick the first bias dimension if exists
            adj = torch.from_numpy(A_pdc[i]).float()
            adj = np.transpose(adj, axes=[1, 2, 0, 3, 4]).reshape(-1, 5, adj.shape[3], adj.shape[4])
            adj = adj.mean(dim=1)
            # edge_index = adj.to_sparse()._indices()

            # features: flatten along all but feature-dim
            x = torch.from_numpy(
                feature_pdc[i].reshape(-1, 30, 5)
            ).float()

            # labels: one per window, so len = x.size(0)
            # y = torch.tensor(label_repeat).long()
            labels.append(torch.tensor(label_repeat).long())
            for j, graph in enumerate(adj):
                num_node = x[j].shape[0]
                edge_index = graph.to_sparse()._indices()
                edge_index, _ = coalesce(edge_index, None, x[j].size(0), x[j].size(0))
                undi_edge_index = torch.unique(edge_index, dim=1)
                # undi_edge_index = remove_self_loops_weights(undi_edge_index)[0]

                row, col = undi_edge_index
                edge_weight = graph[row, col]

                graphs.append(Graph(row, col, edge_weight, num_node, x=x[j], y=None))

            if self.pre_transform:
                data = self.pre_transform(data)
            if self.transform:
                data = self.transform(data)
            # graphs.append(data)
            # labels.append(int(y) if y.numel() == 1 else y)

        return graphs, torch.cat(labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # return a (Data, label) tuple
        return self.graphs[idx], self.labels[idx]


# Example of custom collate_fn to use in DataLoader

def graph_collate(batch):
    graphs, labels = zip(*batch)
    return list(graphs), torch.stack([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in labels])
