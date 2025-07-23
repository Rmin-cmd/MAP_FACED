def add_training_config(parser):
    # seed
    parser.add_argument('--seed', help='seed everything', type=int, default=2024)
    # use gpu
    parser.add_argument('--use_cuda', help='use gpu', type=int, default=1)
    parser.add_argument('--gpu_id', help='gpu id', type=int, default=0)
    parser.add_argument('--normalize_times', help='normalize train', type=int, default=1)
    parser.add_argument('--n_trials', help='n_trials', type=int, default=100)
    # training details
    parser.add_argument('--n_folds', help='number of folds', type=int, default=10)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=200)
    parser.add_argument('--early_stop', help='early stop', type=int, default=100)
    parser.add_argument('--lr', help='learning rate of gnn model', type=float, default=1e-2)
    parser.add_argument('--weight_decay', help='weight decay of gnn model', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', help='training batch size', type=int, default=1000)
    parser.add_argument('--eval_batch_size', help='val and test batch size', type=int, default=1000)
