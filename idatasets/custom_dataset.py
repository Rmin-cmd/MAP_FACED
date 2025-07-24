import os
import torch
import scipy.io as sio
import numpy as np
import h5py
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from utils import load_srt_de  # adapt import to wherever your label loader is


class MeanBandCollapse(object):
    def __call__(self, data: Data) -> Data:
        # data.edge_attr_all: Tensor [B, C, C]
        # 1) mean over B → [C, C]
        A = data.edge_attr_all.mean(dim=0)

        # 2) to sparse
        ei = A.to_sparse()._indices()
        ea = A.to_sparse()._values()

        data.edge_index    = ei
        data.edge_attr     = ea
        # optionally delete the raw bands
        del data.edge_attr_all
        return data


class CustomDataset(InMemoryDataset):
    def __init__(self, args, root, split='train', transform=None, pre_transform=None):
        """
        split: one of 'train', 'val', or 'test' (here val and test use the same fold-split logic)
        args must have:
          - args.n_subs       : total number of subjects
          - args.n_folds      : total number of folds
          - args.fold         : current fold index [0..n_folds-1]
          - args.pdc_path     : path to the .mat or .h5 PDC file
          - args.feature_root_dir : directory containing de_lds_fold{i}.mat
          - args.num_classes  : 2 or 9
        """
        self.args = args
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # one PDC file plus one de_lds_foldX.mat
        return [os.path.basename(self.args.pdc_path),
                f'de_lds_fold{self.args.fold}.mat']

    @property
    def processed_file_names(self):
        return [f'data_fold{self.args.fold}_{self.split}.pt']

    def download(self):
        pass

    def process(self):
        args = self.args
        num_windows = 11

        # 1) Load PDC (connectivity) data
        if self.args.pdc_path.endswith('.h5'):
            with h5py.File(self.args.pdc_path, 'r') as f:
                A_pdc = np.array(f['connectivity'])  # [n_subs, ...]
        else:  # assume .mat
            A_pdc = sio.loadmat(self.args.pdc_path)['data']  # [n_subs, ...]

        # 2) Load features for this fold
        feat_mat = os.path.join(self.args.feature_root_dir,
                                f'de_lds_fold{self.args.fold}.mat')
        feature_pdc = sio.loadmat(feat_mat)['de_lds']  # [n_subs, vids, ch, windows, points]

        # 3) Reshape depending on number of classes
        if args.num_classes == 9:
            label_type = 'cls9'
            # features already in shape [subs, vids, ch, windows, pts]
        elif args.num_classes == 2:
            # select only the 2-class vids and flatten
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

        # 4) Compute repeated labels per window
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

        # 6) Build Data objects
        data_list = []
        for i in tqdm(sel_idx, desc=f"Building {self.split} fold {args.fold}"):
            # adjacency for subject i: assume shape [?, nodes, nodes]
            # Here we pick the first bias dimension if exists
            adj = torch.from_numpy(A_pdc[i]).float()
            adj = np.transpose(adj, axes=[1, 2, 0, 3, 4]).reshape(-1, 5, adj.shape[3], adj.shape[4])
            # edge_index = adj.to_sparse()._indices()
            # edge_index = [adj[:, i, :, :].to_sparse()._indices() for i in range(adj.shape[1])]

            # features: flatten along all but feature-dim
            x = torch.from_numpy(
                feature_pdc[i].reshape(-1, feature_pdc.shape[-1])
            ).float()

            # labels: one per window, so len = x.size(0)
            y = torch.tensor(label_repeat).long()

            data_list.append(Data(x=x, edge_index=None, edge_attr=adj, y=y))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}(fold={self.args.fold}, split={self.split}, ' \
               f'items={self.data.num_graphs})'

    @property
    def train_dataset(self):
        if self.split == 'train':
            # return self
            return CustomDataset(self.args, self.root, split='train',
                             transform=MeanBandCollapse(),
                             pre_transform=self.pre_transform)

    @property
    def val_dataset(self):
        if self.split == 'val':
            # return self
            return CustomDataset(self.args, self.root, split='val',
                             transform=MeanBandCollapse(),
                             pre_transform=self.pre_transform)

    @property
    def test_dataset(self):
        # here test == val
        return self.val_dataset

    @property
    def num_features(self):
        return self.data.x.size(1)

    @property
    def num_classes(self):
        return int(self.data.y.max().item()) + 1
