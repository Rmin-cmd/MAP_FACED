from idatasets.ogb import OGB
from idatasets.pygsd import PyGSDDataset


def load_directed_graph(logger, args, name, root, k, node_split, node_split_id, edge_split, edge_split_id):
    if name.lower() in ("coraml", "citeseerdir", "chameleonfilterdir", "squirrelfilterdir", "wikics", "tolokers", 
                        "amazon-rating", "amazon-computers", "texas", "cornell", "wisconsin", "slashdot", "epinions", 
                        "wikitalk", "roman-empire", "actor"):
        dataset = PyGSDDataset(args, name, root, k, node_split, node_split_id, edge_split, edge_split_id)
    elif name.lower() in ("arxivdir"):
        dataset = OGB(args, name, root, k, node_split, edge_split, edge_split_id)

    if args.heterophily and name.lower() not in ("wikitalk", "slashdot", "epinions"):
        logger.info("Edge homophily: {}, Node homophily: {}, Linkx homophily: {}".format(round(dataset.edge_homophily, 4),
                                                                                       round(dataset.node_homophily, 4),
                                                                                       round(dataset.linkx_homophily, 4)))

    return dataset


def load_graph_dataset(logger, args, name, root):
    if name.lower() in ("graph_dataset"):
        from idatasets.graph_dataset import GraphDataset
        dataset = GraphDataset(args, name, root)
    elif name.lower() in ("custom_dataset"):
        from idatasets.custom_dataset import CustomDataset
        dataset = CustomDataset(args, root)
    else:
        raise NotImplementedError
    return dataset
