import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from tasks.base_task import BaseTask
from tasks.utils import accuracy
from tqdm import tqdm
import torch
from idatasets.custom_dataset import graph_collate


def graph_cls_train(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for i, (data, label) in enumerate(loader):
        # data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # out = model(data)
        out_list = []
        for graph in data:
            # for g in graph:
            out = model.model_forward(graph, device)
            out_list.append(out)
        out = torch.cat(out_list)
        # out = model.model_forward(data, device)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def graph_cls_evaluate(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out_list = []
        for graph in data:
            # for g in graph:
            out = model.model_forward(graph, device)
            out_list.append(out)
        out = torch.cat(out_list)
        # out = model.model_forward(data, device)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


class GraphClassification(BaseTask):
    def __init__(self, logger, train_dataset, val_dataset, test_dataset, model_zoo, normalize_times, lr,
                 weight_decay, epochs, early_stop, device, train_batch_size=32, eval_batch_size=32):
        super(GraphClassification, self).__init__()
        self.logger = logger
        self.normalize_times = normalize_times
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model_zoo = model_zoo
        # self.model_zoo.model_init(self.dataset)
        self.model = self.model_zoo.model_init().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                       collate_fn=graph_collate)
        self.val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=graph_collate)
        self.test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=graph_collate)

        total_epochs_time = []
        two_hundred_epoch_time = []
        total_time = []
        for i in range(self.normalize_times):
            begin_t = time.time()
            if i == 0:
                normalize_times_st = time.time()
            else:
                self.model = self.model_zoo.model_init()
                self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.acc, epochs_time = self.execute()
            total_epochs_time += epochs_time
            two_hundred_epoch_time.append(np.mean(epochs_time) * 200)
            total_time.append(time.time() - begin_t)

        if self.normalize_times > 1:
            logger.info("Optimization Finished!")
            logger.info("Total training time is: {:.4f}s".format(time.time() - normalize_times_st))
            logger.info("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(
                round(np.mean(self.normalize_record["val_acc"]), 2),
                round(np.std(self.normalize_record["val_acc"], ddof=1), 2),
                round(np.mean(self.normalize_record["test_acc"]), 2),
                round(np.std(self.normalize_record["test_acc"], ddof=1), 2)))
            logger.info(
                "Mean Epoch ± Std Epoch: {:.4f}s±{:.4f}s, Mean 200 Epoch ± Std Epoch: {:.4f}s±{:.4f}s, Mean Total ± Std Total: {:.4f}s±{:.4f}s".format(
                    np.mean(total_epochs_time), np.std(total_epochs_time),
                    np.mean(two_hundred_epoch_time), np.std(two_hundred_epoch_time),
                    np.mean(total_time), np.std(total_time)))

    def execute(self):
        self.model = self.model.to(self.device)

        best_val = 0.
        best_test = 0.
        stop = 0
        time_list = []

        t_total = time.time()
        for epoch in tqdm(range(self.epochs)):
            if stop > self.early_stop:
                self.logger.info("Early stop!")
                break
            t = time.time()

            loss_train = graph_cls_train(self.model, self.train_loader, self.device, self.optimizer, self.criterion)
            acc_val = graph_cls_evaluate(self.model, self.val_loader, self.device)
            acc_test = graph_cls_evaluate(self.model, self.test_loader, self.device)

            epoch_time = time.time() - t
            if self.normalize_times == 1:
                self.logger.info("Epoch: {:03d}, loss_train: {:.4f}, acc_val: {:.4f}, "
                                 "acc_test: {:.4f}, time: {:.4f}s".format(epoch + 1, loss_train, acc_val,
                                                                          acc_test, epoch_time))
            if epoch != 0:
                time_list.append(epoch_time)
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                stop = 0
            stop += 1

        if self.normalize_times == 1:
            self.logger.info("Optimization Finished!")
            self.logger.info("Total training time is: {:.4f}s".format(time.time() - t_total))
            self.logger.info(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')

        return best_test, time_list
