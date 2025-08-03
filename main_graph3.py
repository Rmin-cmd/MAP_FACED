# main.py
import argparse
import datetime
import os
import time
import warnings

import numpy as np
import torch

from configs.data_config import add_data_config
from configs.model_config import add_model_config
from configs.training_config import add_training_config
from idatasets.load_data import load_graph_dataset
from logger import Logger
from models.model_init import ModelZoo
from tasks.graph_classification import GraphClassification
from utils import seed_everything, get_params
from idatasets.custom_dataset import CustomDataset

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    add_data_config(parser)       # must include args.n_folds
    add_model_config(parser)
    add_training_config(parser)
    # Make sure data_config adds something like:
    #   parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    # Setup overall logging path
    now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("log", "graph", args.model_name, args.graph_data_name)
    os.makedirs(log_dir, exist_ok=True)
    logger_name = os.path.join(log_dir, now_time + ".log")
    logger = Logger(logger_name)
    logger.info(f"Program start: {now_time}")
    logger.info(f"Using seed: {args.seed} — CUDA: {args.use_cuda}, GPU ID: {args.gpu_id}")

    # Set seeds & device
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    )

    # We'll collect best test accuracies across folds
    test_acc_folds = []
    start_fold = 0

    if args.resume_from_checkpoint:
        for fold in range(args.n_folds):
            checkpoint_path = os.path.join(log_dir, f"checkpoint_fold_{fold}.pth")
            if os.path.exists(checkpoint_path):
                # This fold is considered complete, load its accuracy and skip
                checkpoint = torch.load(checkpoint_path)
                test_acc_folds.append(checkpoint['best_test_acc'])
                logger.info(f"Fold {fold} already completed. Loaded test accuracy: {checkpoint['best_test_acc']:.4f}")
                start_fold = fold + 1
            else:
                # This is the first incomplete fold
                logger.info(f"Resuming training from fold {fold}")
                break

    for fold in range(start_fold, args.n_folds):
        logger.info(f"\n=== Fold {fold+1}/{args.n_folds} ===")
        args.fold = fold

        # --- 1) Load dataset for this fold ---
        t0 = time.time()
        # The dataset object now represents a single fold
        # dataset = load_graph_dataset(logger, args, name=args.graph_data_name, root=args.data_root)
        t1 = time.time()
        # logger.info(f"Loaded Fold {fold} dataset in {t1-t0:.2f}s; "
        #             f"#features={dataset.num_features}, #classes={dataset.num_classes}")

        # The CustomDataset class now handles the train/val/test split internally
        train_dataset = CustomDataset(args, root=args.data_root, split='train')
        val_dataset = CustomDataset(args, root=args.data_root, split='val')
        test_dataset = CustomDataset(args, root=args.data_root, split='val')

        # --- 2) Init model & task runner ---
        # model_zoo = ModelZoo(logger, args, 30, dataset.num_features, dataset.num_classes, None, "graph")
        model_zoo = ModelZoo(logger, args, 30, 5, 9, None, "graph")
        task = GraphClassification(
            logger,
            train_dataset,  # Pass the training dataset
            val_dataset,    # Pass the validation dataset
            test_dataset,   # Pass the test dataset
            model_zoo,
            normalize_times=args.normalize_times,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.num_epochs,
            early_stop=args.early_stop,
            device=device,
            args=args,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size
        )

        # --- 3) Run training/eval for this fold ---
        # GraphClassification will log per-epoch info and return best test ACC
        best_test_acc, _ = task.execute()
        test_acc_folds.append(best_test_acc)
        logger.info(f"Fold {fold}: best test accuracy = {best_test_acc:.4f}")

    # --- Summarize across folds ---
    mean_acc = np.mean(test_acc_folds)
    std_acc  = np.std(test_acc_folds, ddof=1) if args.n_folds > 1 else 0.0
    logger.info(f"\n=== K‑Fold Summary ({args.n_folds} folds) ===")
    logger.info(f"Test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"\nFinal: {mean_acc:.4f} ± {std_acc:.4f}")

    # Optionally save the summary to disk
    summary_path = os.path.join(log_dir, "kfold_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Folds test acc: {test_acc_folds}\n")
        f.write(f"Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}\n")
