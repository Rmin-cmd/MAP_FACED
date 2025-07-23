import argparse
import datetime
import os
import time
import warnings

import torch

from configs.data_config import add_data_config
from configs.model_config import add_model_config
from configs.training_config import add_training_config
from idatasets.load_data import load_directed_graph
from logger import Logger
from models.model_init import ModelZoo
from tasks.node_classification import NodeClassification
from utils import seed_everything, get_params

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    add_data_config(parser)
    add_model_config(parser)
    add_training_config(parser)
    args = parser.parse_args()

    dataset_name = args.data_name
    model_name = args.model_name

    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join("log", model_name, dataset_name, args.data_node_split)
    logger_name = os.path.join(log_dir, str(now_time)[:str(now_time).find('.')] + ".log")
    logger = Logger(logger_name)

    logger.info(f"program start: {now_time}")

    # set up seed
    logger.info(f"random seed: {args.seed}")
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

    # set up idatasets
    set_up_datasets_start_time = time.time()
    logger.info(f"Load unsigned & directed & unweighted network: {args.data_name}")
    dataset = load_directed_graph(logger, args, name=args.data_name, root=args.data_root, k=args.data_dimension_k,
                                  node_split=args.data_node_split, edge_split=args.data_edge_split,
                                  node_split_id=args.data_node_split_id, edge_split_id=args.data_edge_split_id)
    set_up_datasets_end_time = time.time()

    logger.info(f"datasets: {args.data_name}, root dir: {args.data_root}, node-level split method: "
                f"{args.data_node_split}, id: {args.data_node_split_id}, "
                f"edge-level split method: {args.data_edge_split}, id: {args.data_edge_split_id}, "
                f"the running time is: {round(set_up_datasets_end_time-set_up_datasets_start_time,4)}s")
    logger.info(f"num_epochs: {args.num_epochs}, early_stop: {args.early_stop}, lr: {args.lr}, weight_decay: {args.weight_decay}")
    logger.info(f"dataset.x.shape: {dataset.x.shape}")
    label_info = max(dataset.y)+1 if dataset.y is not None else -1
    logger.info(f"max(dataset.y)+1: {label_info}")
    logger.info(f"dataset.num_node: {dataset.num_node}")
    logger.info(f"Real edges -> dataset.num_edge: {dataset.num_edge}")
    logger.info(f"min(dataset.adj.data): {min(dataset.adj.data)}")
    logger.info(f"max(dataset.adj.data): {max(dataset.adj.data)}")
    logger.info(f"dataset.adj.data: {dataset.adj.data}")

    if args.data_name not in ("wikitalk", "slashdot", "epinions"):
        model_zoo = ModelZoo(logger, args, dataset.num_node, dataset.num_features, dataset.num_node_classes, dataset.y, dataset.test_idx, "node")
        run = NodeClassification(logger, dataset, model_zoo, normalize_times=args.normalize_times, lr=args.lr, device=device,
                                 weight_decay=args.weight_decay, epochs=args.num_epochs, early_stop=args.early_stop)
        logger.info("# NodeClassification Params:" + str(get_params(model_zoo.model_init())))
    else:
        raise NotImplementedError
