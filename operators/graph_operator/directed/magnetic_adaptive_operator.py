from operators.utils import adj_to_directed_symmetric_map_norm


class MagAdaptiveGraphOp:
    def __init__(self, q):
        self.q = q

    def construct_adj(self, adj):
        adj = adj.tocoo()
        return adj_to_directed_symmetric_map_norm(adj, self.q)
        