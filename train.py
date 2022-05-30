import os
import copy
import time
from cv2 import split
import tqdm
import json
import argparse
import numpy as np
import torch
from torch import nn
from torch import cuda
from torch import optim
from torch import Tensor
from torch.utils import data
import torchvision
from torchvision import models
from torchvision import datasets
from torchvision import transforms

from args import get_args


def load_model(num_class, device='cpu'):
    model: nn.Module = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_class)

    if device == 'cuda':
        model = nn.parallel.DataParallel(model)
        model.to('cuda')
    elif device[:5] == 'cuda:':
        device_ids = [int(id) for id in device[5:].split(',')]
        model = nn.parallel.DataParallel(model, device_ids=device_ids)
        model.to('cuda')
    elif device == 'cpu':
        model.to('cpu')
    else:
        raise ValueError('Invalid device: %s' % device)
    print('Model run in device: \033[01;32m%s\033[0m' % device)
    return model


def load_dataset(): pass


def main():
    args = get_args()
    print(chr(128640), args)
    DEVICE: str = args.device
    BATCH_SIZE: int = args.batch_size
    NUM_WORKER: int = args.num_worker
    MAX_EPOCH: int = args.max_epoch
    LR: float = args.lr

    model = load_model(num_class=10, device=args.device)


if __name__ == '__main__':
    main()
