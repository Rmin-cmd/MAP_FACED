import torch
import numpy as np
import os.path as osp
import pickle as pkl
import pandas as pd

from idatasets.base_data import Graph
from idatasets.base_dataset import NodeDataset
from idatasets.link_split import link_class_split
from ogb.nodeproppred import PygNodePropPredDataset
from idatasets.utils import pkl_read_file, remove_self_loops, to_undirected, edge_homophily, node_homophily, linkx_homophily, set_spectral_adjacency_reg_features

class OGB(NodeDataset):
    '''
    Dataset description: (Open Graph Benchmark): https://ogb.stanford.edu/docs/nodeprop/
    Directed infomation:  Directed network (ogbn-arxivdir) -> notably, in this directed setting, we implement it as an directed graph.

    -> ogbn-arxivdir:     unsigned & directed & unweighted homogeneous simplex network    

    We remove all multiple edges and self-loops from the original dataset. 
    
    ogbn-arxivdir:      The ogbn-arxivdir dataset is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers.
                        169,343 nodes, 1,166,243 edges, 128 feature dimensions, 40 classes num.
                        Notably, we load the original idatasets (directed graph) hence consistent with the results reported in the paper.
                        NeurIPS'21 Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, LINKX https://arxiv.org/pdf/2110.14446.pdf
    
    split:
        ogbn-arxivdir:
            official:   We propose to train on papers published until 2017, 
                        validate on those published in 2018, 
                        and test on those published since 2019.
                        train/val/test = 90,941/29,799/48,603

    '''
    def __init__(self, args, name="arxivdir", root="./idatasets/", k=2,
                 node_split="official", edge_split="direction", edge_split_id=0):
        name = name.lower()
        if name not in ["arxivdir"]:
            raise ValueError("Dataset name not found!")
        super(OGB, self).__init__(root, name, k)

        self.read_file()
        self.edge_split = edge_split
        self.edge_split_id = edge_split_id
        self.cache_edge_split = osp.join(self.raw_dir, "{}-edge-splits".format(self.name))
        self.train_idx, self.val_idx, self.test_idx = self.generate_split(node_split)
 
        edge_index = torch.from_numpy(np.vstack((self.edge.row.numpy(), self.edge.col.numpy()))).long()
        self.observed_edge_idx, self.observed_edge_weight, self.train_edge_pairs_idx, self.val_edge_pairs_idx, self.test_edge_pairs_idx, self.train_edge_pairs_label, self.val_edge_pairs_label, self.test_edge_pairs_label\
        = link_class_split(edge_index=edge_index, A=self.edge.sparse_matrix,
                        cache_edge_split=self.cache_edge_split, 
                        task=self.edge_split, edge_split_id=self.edge_split_id,
                        prob_val=0.15, prob_test=0.05, )
        self.num_node_classes = self.num_classes
        if edge_split in ("existence", "direction", "sign"):
            self.num_edge_classes = 2
        elif edge_split in ("three_class_digraph"):
            self.num_edge_classes = 3
        elif edge_split in ("four_class_signed_digraph"):
            self.num_edge_classes = 4
        elif edge_split in ("five_class_signed_digraph"):
            self.num_edge_classes = 5
        else:
            self.num_edge_classes = None
        if args.heterophily:
            self.edge_homophily = edge_homophily(self.adj, self.y)
            self.node_homophily = node_homophily(self.adj, self.y)
            self.linkx_homophily = linkx_homophily(self.adj, self.y)
        
    @property
    def raw_file_paths(self):
        if self.name == "arxivdir":
            name = "arxiv"
        filepath = "ogbn_" + name + "/processed/geometric_data_processed.pt"
        return osp.join(self.raw_dir, filepath)

    @property
    def processed_file_paths(self):
        filename = "graph"
        return osp.join(self.processed_dir, "{}.{}".format(self.name, filename))

    def read_file(self):
        self.data = pkl_read_file(self.processed_file_paths)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        self.num_features = self.data.num_features
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge
        # import time
        # t1 = time.time()
        # edge_weight = self.edge.edge_weight
        # indices = torch.vstack((self.edge.row, self.edge.col)).long()
        # edge_num_node = indices.max().item() + 1
        # features = set_spectral_adjacency_reg_features(edge_num_node, indices, edge_weight)
        # logger.info(time.time()-t1)

    def download(self):
        if self.name == "arxivdir":
            name = "arxiv"
        PygNodePropPredDataset("ogbn-" + name, self.raw_dir)

    def process(self):
        if self.name == "arxivdir":
            name = "arxiv"
        dataset = PygNodePropPredDataset("ogbn-" + name, self.raw_dir)

        data = dataset[0]
        features, labels = data.x.numpy().astype(np.float32), data.y.to(torch.long).squeeze(1)
        num_node = data.num_nodes

        if self.name == "arxivdir":
            undi_edge_index = torch.unique(data.edge_index, dim=1)
        
        row, col = undi_edge_index
        edge_weight = torch.ones(len(row))

        g = Graph(row, col, edge_weight, num_node, x=features, y=labels)
        with open(self.processed_file_paths, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

    def generate_split(self, split):
        if split == "official":
            if self.name == "arxivdir":
                root = "data/arxivdir/raw/ogbn_arxiv"
                split_type = "time"
                
            path = osp.join(root, 'split', split_type)
            train_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            val_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)
            test_idx = torch.from_numpy(pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]).to(torch.long)

        elif split == "random":
            raise NotImplementedError
        
        else:
            raise ValueError("Please input valid split pattern!")

        return train_idx, val_idx, test_idx
