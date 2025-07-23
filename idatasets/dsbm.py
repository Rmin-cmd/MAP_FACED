import os
import math
import torch
import numpy as np
import pickle as pkl
import os.path as osp
import networkx as nx
import scipy.sparse as sp
import numpy.random as rnd

from datasets.base_data import Graph
from datasets.utils import file_exist
from datasets.link_split import link_class_split
from datasets.node_split import node_class_split
from datasets.utils import pkl_read_file, edge_homophily, node_homophily, linkx_homophily, set_spectral_adjacency_reg_features


class DSBM():
    """A directed stochastic block model graph generator from the
    DIGRAC: Digraph Clustering Based on Flow Imbalance: https://arxiv.org/pdf/2106.05194.pdf.

    Arg types:

        num_node - Number of nodes.
        clusters - Number of clusters.
        sparsity - Sparsity value, edge probability.
        meta_pattern - The meta-graph adjacency matrix to generate edges.
        size_ratio - The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
    """
    def __init__(self, logger, args, num_node, clusters, sparsity, meta_pattern, k=2, sbm_noise=0.5, size_ratio=1,
                 node_split="official", node_split_id=0, edge_split="direction", edge_split_id=0):
        self.processed_dir = osp.join("./idatasets/directed/synthetic/", f"{num_node}_{clusters}_{sparsity}_{meta_pattern}_{sbm_noise}_{size_ratio}")
        self.processed_file_path = osp.join(self.processed_dir, "synthetic.graph") 

        self.k = k
        self.num_node = num_node
        self.clusters = clusters
        self.sparsity = sparsity
        self.meta_pattern = meta_pattern
        self.sbm_noise = sbm_noise
        self.size_ratio = size_ratio

        self.node_split = node_split
        self.node_split_id = node_split_id
        self.edge_split = edge_split
        self.edge_split_id = edge_split_id
        self.official_split = None
        self.cache_node_split = osp.join(self.processed_dir, "node-splits")
        self.cache_edge_split = osp.join(self.processed_dir, "edge-splits")

        if file_exist(self.processed_dir):
            pass
        else:
            if not file_exist(self.processed_dir):
                os.makedirs(self.processed_dir)
        try: self.read()
        except: 
            self.process()
            self.read()

        self.train_idx, self.val_idx, self.test_idx, self.seed_idx, self.stopping_idx = node_class_split(name=None, data=self.data, 
                                                                                    cache_node_split=self.cache_node_split,
                                                                                    official_split=self.official_split,
                                                                                    split=self.node_split, node_split_id=self.node_split_id, 
                                                                                    train_size_per_class=200, val_size=2000)
        
        edge_index = torch.from_numpy(np.vstack((self.edge.row.numpy(), self.edge.col.numpy()))).long()
        self.observed_edge_idx, self.observed_edge_weight, self.train_edge_pairs_idx, self.val_edge_pairs_idx, self.test_edge_pairs_idx, self.train_edge_pairs_label, self.val_edge_pairs_label, self.test_edge_pairs_label\
        = link_class_split(edge_index=edge_index, A=self.edge.sparse_matrix,
                        cache_edge_split=self.cache_edge_split, 
                        task=self.edge_split, edge_split_id=self.edge_split_id,
                        prob_val=0.05, prob_test=0.05, )
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

        if args.heterogeneity and self.name not in ("wikitalk", "slashdot", "epinions"):
            self.edge_homophily = edge_homophily(self.adj, self.y)
            self.node_homophily = node_homophily(self.adj, self.y)
            self.linkx_homophily = linkx_homophily(self.adj, self.y)
            logger.info("Edge homophily: {}, Node homophily:{}, Linkx homophily:{}".format(round(self.edge_homophily, 4),
                                                                                        round(self.node_homophily, 4),
                                                                                        round(self.linkx_homophily,4)))
            
    def read(self):
        self.data = pkl_read_file(self.processed_file_path)
        self.edge = self.data.edge
        self.node = self.data.node
        self.x = self.data.x
        self.y = self.data.y
        self.adj = self.data.adj
        self.num_features = self.data.num_features 
        self.num_classes = self.data.num_classes
        self.num_node = self.data.num_node
        self.num_edge = self.data.num_edge

    def process(self):
        print("Generate synthetic data, it may take a while...")
        size = [0] * self.clusters
        perm = rnd.permutation(self.num_node)
        if self.size_ratio > 1:
            ratio_each = np.power(self.size_ratio, 1/(self.clusters-1))
            smallest_size = math.floor(
                self.num_node*(1-ratio_each)/(1-np.power(ratio_each, self.clusters)))
            size[0] = smallest_size
            if self.clusters > 2:
                for i in range(1, self.clusters-1):
                    size[i] = math.floor(size[i-1] * ratio_each)
            size[self.clusters-1] = self.num_node - np.sum(size)
        else:  # degenerate case, equaivalent to 'uniform' sizes
            size = [math.floor((i + 1) * self.num_node / self.clusters) -
                    math.floor((i) * self.num_node / self.clusters) for i in range(self.clusters)]
        labels = []
        for i, s in enumerate(size):
            labels.extend([i]*s)
        labels = np.array(labels)
        # permutation
        labels = labels[perm]
        meta_adj = generate_meta_adj(self.meta_pattern, self.clusters, self.sbm_noise)

        g = nx.stochastic_block_model(sizes=size, p=self.sparsity*meta_adj, directed=True)
        A = nx.adjacency_matrix(g)[perm][:, perm].tocoo()
        row, col = A.row, A.col
        edge_index = np.vstack((row, col))
        edge_index = torch.from_numpy(edge_index).long()
        row, col = edge_index
        edge_num_node = edge_index.max().item() + 1
        num_node = edge_num_node
        edge_weight = torch.ones(len(row))
        features = set_spectral_adjacency_reg_features(edge_num_node, edge_index, edge_weight, self.k)
        labels = torch.tensor(labels, dtype=torch.long)

        g = Graph(row, col, edge_weight, num_node, x=features, y=labels)

        with open(self.processed_file_path, 'wb') as rf:
            try:
                pkl.dump(g, rf)
            except IOError as e:
                print(e)
                exit(1)

def generate_meta_adj(meta_pattern, clusters, sbm_noise):
    meta_adj = np.zeros((clusters, clusters), dtype=float)
    if meta_pattern == "cycle":
        for i in range(clusters):
            for j in range(clusters):
                part1 = int((j == ((i + 1) % clusters)))
                part2 = int((j == ((i - 1) % clusters)))
                part3 = int((j == i))
                meta_adj[i][j] = (1 - sbm_noise) * part1 + sbm_noise * part2 + 0.5 * part3
    
    elif meta_pattern == "path":
        for i in range(clusters):
            for j in range(clusters):
                part1 = int((j == i + 1))
                part2 = int((j == i - 1))
                part3 = int((j == i))
                meta_adj[i][j] = (1 - sbm_noise) * part1 + sbm_noise * part2 + 0.5 * part3

    elif meta_pattern == "complete":
        for i in range(clusters):
            for j in range(clusters):
                if i == j:
                    meta_adj[i][j] = 0.5
                elif i < j:
                    meta_adj[i][j] = sbm_noise
                else:
                    meta_adj[i][j] = 1 - sbm_noise
    
    elif meta_pattern == "star":
        center = math.floor((clusters - 1) / 2)
        for i in range(clusters):
            for j in range(clusters):
                part1 = int((i == center) and (j & 1 != 0))
                part2 = int((i == center) and (j & 1 == 0))
                part3 = int((j == center) and (j & 1 != 0))
                part4 = int((j == center) and (j & 1 == 0))
                meta_adj[i][j] = (1 - sbm_noise) * part1 + sbm_noise * part2 +\
                                (1 - sbm_noise) * part3 + sbm_noise * part4

    else:
        raise ValueError(f"{meta_pattern} is not supported!")
    
    return meta_adj


