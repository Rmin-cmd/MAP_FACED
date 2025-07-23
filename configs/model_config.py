def add_model_config(parser):
    # directed graph
    #   (baseline-spectral) map mapplus
    parser.add_argument('--model_name', help='gnn model', type=str, default="mapplus")
    # magnetic adaptive propagation plus
    parser.add_argument('--num_layers', help='number of gnn layers', type=int, default=1)
    parser.add_argument('--dropout', help='drop out of gnn model', type=float, default=0.5)
    parser.add_argument('--hidden_dim', help='hidden units', type=int, default=128)
    parser.add_argument('--edge_dim', help='hidden units of linear-based model in edge-level tasks', type=int, default=64)
    parser.add_argument('--prop_steps', help='prop steps', type=int, default=2)
    parser.add_argument('--r', help='symmetric normalized unit', type=float, default=0.5)
    parser.add_argument('--node_q', help='the imaginary part of the complex unit in node-level tasks', type=float, default=0.05)
    parser.add_argument('--use_att', help='whether use attention', type=int, default=1)
