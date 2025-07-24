import argparse
import datetime
import os
import time
import warnings

import torch

from configs.data_config import add_data_config
from configs.model_config import add_model_config
from configs.training_config import add_training_config
from idatasets.load_data import load_graph_dataset
from logger import Logger
from models.model_init import ModelZoo
from tasks.graph_classification import GraphClassification
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

    log_dir = os.path.join("log", "graph", model_name, dataset_name)
    logger_name = os.path.join(log_dir, str(now_time)[:str(now_time).find('.')] + ".log")
    logger = Logger(logger_name)

    logger.info(f"program start: {now_time}")

    # set up seed
    logger.info(f"random seed: {args.seed}")
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if (args.use_cuda and torch.cuda.is_available()) else 'cpu')

    # set up idatasets
    set_up_datasets_start_time = time.time()
    logger.info(f"Load graph dataset: {args.graph_data_name}")
    dataset = load_graph_dataset(logger, args, name=args.graph_data_name, root=args.data_root)
    set_up_datasets_end_time = time.time()

    logger.info(f"datasets: {args.data_name}, root dir: {args.data_root}, "
                f"the running time is: {round(set_up_datasets_end_time - set_up_datasets_start_time, 4)}s")
    logger.info(
        f"num_epochs: {args.num_epochs}, early_stop: {args.early_stop}, lr: {args.lr}, weight_decay: {args.weight_decay}")

    model_zoo = ModelZoo(logger, args, None, dataset.num_features, dataset.num_classes, None, None, "graph")
    run = GraphClassification(logger, dataset, model_zoo, normalize_times=args.normalize_times, lr=args.lr,
                              device=device,
                              weight_decay=args.weight_decay, epochs=args.num_epochs, early_stop=args.early_stop,
                              train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)
    logger.info("# GraphClassification Params:" + str(get_params(model_zoo.model_init())))
