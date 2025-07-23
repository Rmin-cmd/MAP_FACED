def add_data_config(parser):
    parser.add_argument('--heterophily', help='Select whether to analyze network heterophily', type=bool, default=False)
    # unsigned & directed & unweighted data
    # root: ./idatasets
    # node split: official (Random splitting of fixed rates with official split files)
    # node split id:
    #             coraml & citeseerdir & chameleondir & squirreldir: 0-9
    #             wikics: 0-19
    # edge split: existence & direction & three_class_digraph
    # edge split id: 0-9
    parser.add_argument('--data_name', help='unsigned & directed & unweighted data name', type=str, default="coraml")
    parser.add_argument('--data_root', help='unsigned & directed & unweighted data root', type=str, default="data/")
    parser.add_argument('--data_node_split', help='unsigned & directed & unweighted data node split method', type=str, default="official")
    parser.add_argument('--data_edge_split', help='unsigned & directed & unweighted data edge split method', type=str, default="direction")
    parser.add_argument('--data_node_split_id', help='unsigned & directed & unweighted data node split id', type=int, default=0)
    parser.add_argument('--data_edge_split_id', help='unsigned & directed & unweighted data edge split id', type=int, default=0)
    parser.add_argument('--data_dimension_k', help='generate node features based on the structure topology', type=int, default=100)
