import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn *= s
        pp += nn
    return pp


def load_srt_de(data, channel_norm,label_type, num_windows):
    # isFilt: False  filten:1   channel_norm: True
    n_subs = 123
    if label_type == 'cls2':
        n_vids = 24
    elif label_type == 'cls9':
        n_vids = 28
    n_samples = np.ones(n_vids).astype(np.int32) * num_windows  # (30,30,...,30)

    n_samples_cum = np.concatenate(
        (np.array([0]), np.cumsum(n_samples)))  # (0,30,60,...,810,840)

    # Normalization for each sub
    if channel_norm:
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - np.mean(data[i, :, :],
                             axis=0)) / np.std(data[i, :, :], axis=0)

    if label_type == 'cls2':
        label = [0] * 12
        label.extend([1] * 12)
    elif label_type == 'cls9':
        label = [0] * 3
        for i in range(1, 4):
            label.extend([i] * 3)
        label.extend([4] * 4)
        for i in range(5, 9):
            label.extend([i] * 3)
        # print(label)

    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return label_repeat

