import numpy as np
import torch

from sklearn.metrics import roc_auc_score, f1_score


def accuracy(output, labels, return_idx=False):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    if not return_idx:
        return (correct.sum() / len(labels) * 100.0).item()
    else:
        return (correct.sum() / len(labels) * 100.0).item(), np.where(correct.cpu()==1)


def auc(output, labels):
    output = output.detach().cpu().numpy()
    # output[np.where(np.isnan(output) == True)] = 0
    minus = np.max(output, axis=1).reshape((output.shape[0],1))
    exp = np.exp(output - minus)
    y_score = exp / np.sum(exp, axis=1).reshape((exp.shape[0], 1))
    try:
        score = roc_auc_score(labels.cpu(), y_score, multi_class='ovr')
    except:
        score = roc_auc_score(labels.cpu(), y_score[:, 1])
    return score * 100.0


def macro_f1(output, labels):
    pred = output.max(1)[1].type_as(labels)
    return f1_score(labels.cpu(), pred.cpu(), average='macro') * 100.0


def node_cls_train(model, train_idx, labels, device, optimizer, loss_fn, loss_weight):
    model.train()
    optimizer.zero_grad()
    if loss_weight is not None:
        train_output, contrastive_loss = model.model_forward(train_idx, device, train_flag=1)
        loss_train = loss_fn(train_output, labels[train_idx]) + loss_weight * contrastive_loss
    else:
        train_output = model.model_forward(train_idx, device)
        loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train


def node_cls_mini_batch_train(model, train_idx, train_loader, labels, device, optimizer, loss_fn, loss_weight):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        if loss_weight is not None:
            train_output, contrastive_loss = model.model_forward(train_idx, device, train_flag=1)
            loss_train = loss_fn(train_output, labels[train_idx]) + loss_weight * contrastive_loss
        else:
            train_output = model.model_forward(train_idx, device)
            loss_train = loss_fn(train_output, labels[train_idx])
        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_idx)

    return loss_train, acc_train.item()


def node_cls_evaluate(model, val_idx, test_idx, labels, device, postpro):
    model.eval()
    val_output = model.model_forward(idx=val_idx, device=device)
    test_output = model.model_forward(idx=test_idx, device=device)
    acc_val = accuracy(val_output, labels[val_idx])
    if postpro:
        acc_test, idx = accuracy(test_output, labels[test_idx], postpro)
        return acc_val, acc_test, test_idx[idx]
    else:
        acc_test = accuracy(test_output, labels[test_idx], postpro)
        return acc_val, acc_test


def node_cls_mini_batch_evaluate(model, val_idx, val_loader, test_idx, test_loader, labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_output = model.model_forward(batch, device)
        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_idx)

    for batch in test_loader:
        test_output = model.model_forward(batch, device)
        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_idx)

    return acc_val.item(), acc_test.item()
