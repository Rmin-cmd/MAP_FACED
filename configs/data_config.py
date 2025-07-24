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
    parser.add_argument("--n_subs", type=int, default=123, help= "number of subjects")
    parser.add_argument('--data_name', help='unsigned & directed & unweighted data name', type=str, default="coraml")
    parser.add_argument('--graph_data_name', help='graph classification dataset name', type=str, default="custom_dataset")
    parser.add_argument('--data_root', help='unsigned & directed & unweighted data root', type=str, default="data/")
    parser.add_argument('--feature_root_dir', help='DE features of brain connectivity', type=str, default="data/features")
    parser.add_argument('--pdc_path', help='PDC Effective Brain Connectivity', type=str, default="data/connectivity.mat")
    parser.add_argument('--num_classes', help="number of available classes", type=int, default=9)
    parser.add_argument('--data_node_split', help='unsigned & directed & unweighted data node split method', type=str, default="official")
    parser.add_argument('--data_edge_split', help='unsigned & directed & unweighted data edge split method', type=str, default="direction")
    parser.add_argument('--data_node_split_id', help='unsigned & directed & unweighted data node split id', type=int, default=0)
    parser.add_argument('--data_edge_split_id', help='unsigned & directed & unweighted data edge split id', type=int, default=0)
    parser.add_argument('--data_dimension_k', help='generate node features based on the structure topology', type=int, default=100)
