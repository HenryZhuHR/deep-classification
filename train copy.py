import os
import copy
import time
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
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------------
#   Args
# --------------------------------------------------------
parser = argparse.ArgumentParser(description='train model')
# base train argument
parser.add_argument('--arch', type=str,default='')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int,
                    help='random seed set')
parser.add_argument('--data', type=str, default='%s/datasets/custom' % os.path.expanduser('~'),
                    help='dataset folder')
# model save argument
parser.add_argument('--model_save_dir', type=str, default='server/checkpoints')
parser.add_argument('--model_save_name', type=str,
                    help='using arch name if not given')
parser.add_argument('--logdir', type=str, default='server',
                    help='train log save folder')
parser.add_argument('--model_summary', action='store_true',
                    help='if print model summary')

parser.add_argument('--epsilon', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--iters', type=int)
args = parser.parse_args()

# Parse Args
ARCH: str = args.arch
DEVICE: str = args.device
BATCH_SIZE: int = args.batch_size    # 2
NUM_WORKERS: int = args.num_worker   # 0
MAX_EPOCH: int = args.max_epoch    # 100
LR: float = args.lr   # 0.01
SEED: int = args.seed
DATASET_DIR: str = args.data  # 'data'

MODEL_SAVE_DIR: str = args.model_save_dir  # 'checkpoints'
MODEL_SAVE_NAME: str = ARCH if args.model_save_name == None else args.model_save_name  # 'NONE'/ARCH
LOG_DIR: str = args.logdir    # 'where tensorboard data save (runs)'
IS_MODEL_SUMMARY: bool = args.model_summary

EPSILON: float = args.epsilon/255
ALPHA: float = args.alpha
ITERS: float = args.iters
print()
print("-- Attack Parameters: ")
print(" %s epsilon: " % chr(128296), EPSILON)
print(" %s alpha  : " % chr(128296), ALPHA)
print(" %s iters  : " % chr(128296), ITERS)
print()



if __name__ == '__main__':
    start_time = time.strftime("%m%d_%H%M", time.localtime())

    # create train log save folder
    os.makedirs('%s/%s' % (LOG_DIR, ARCH), exist_ok=True)

    # init Tensorborad SummaryWriter

    writer = SummaryWriter('%s/%s/%s' % (LOG_DIR, ARCH, MODEL_SAVE_NAME))

    # ----------------------------------------
    #   Load dataset
    # ----------------------------------------
    DATA_TRANSFORM = {
        'train': transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        'valid': transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        'test': transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    }
    train_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'),
                                     transform=DATA_TRANSFORM['train'])
    valid_set = datasets.ImageFolder(os.path.join(DATASET_DIR, 'valid'),
                                     transform=DATA_TRANSFORM['valid'])
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    num_class = len(train_set.classes)
    print('%s Load \033[0;32;40m%d\033[0m classes dataset' %
          (chr(128229), num_class))

    # save class json file in log_dir
    with open(os.path.join(LOG_DIR, 'class_indices.json'), 'w') as f:
        f.write(json.dumps(
            {value: key for key, value in train_set.class_to_idx.items()},
            indent=4
        ))

    # ----------------------------------------
    #   Load model and fine tune
    # ----------------------------------------
    print('%s Try to load model \033[0;32;40m%s\033[0m ...' % (
        chr(128229), ARCH))
    from torchvision import models
    model: nn.Module = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.MSELoss()
    loss_function.to(DEVICE)

    # ----------------------------------------
    #   tensorboard :   Add model graph
    #   torchsummary:   Summary model
    # ----------------------------------------
    input_tensor_sample: Tensor = train_set[0][0]
    writer.add_graph(model, input_to_model=(
        input_tensor_sample.unsqueeze(0)).to(DEVICE))
    if IS_MODEL_SUMMARY:
        try:
            from torchsummary import summary
        except:
            print('please install torchsummary by command: pip instsll torchsummary')
        else:
            print(summary(model, input_tensor_sample.size(),
                  device=DEVICE.split(':')[0]))

    # ----------------------------------------
    #   set train random seed
    # ----------------------------------------
    if SEED is not None:
        torch.manual_seed(SEED)  # set seed for current CPU
        torch.cuda.manual_seed(SEED)  # set seed for current GPU
        torch.cuda.manual_seed_all(SEED)  # set seed for all GPU

    # ----------------------------------------
    #   Train model
    # ----------------------------------------

    print('%s Train model in device: \033[0;32;40m%s\033[0m ' % (
        chr(128640), DEVICE))
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    train_log = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    for epoch in range(1, MAX_EPOCH + 1):
        print('\033[0;32;40m[train: %s]\033[0m' % ARCH, end=' ')
        print('[Epoch] %d/%d' % (epoch, MAX_EPOCH), end=' ')
        print('[Batch Size] %d' % (BATCH_SIZE), end=' ')
        print('[LR] %f' % (LR))

        # --- train ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0    # how many data has trained
        model.train()
        pbar = tqdm.tqdm(train_loader)
        # mini batch
        is_recon = 0
        for images, labels in pbar:
            images: Tensor = images.to(DEVICE)
            labels: Tensor = labels.to(DEVICE)
            batch = images.size(0)
            num_data += batch


            output: Tensor = model(images.to(DEVICE))
            _, pred = torch.max(output, 1)
            loss: Tensor = loss_function(output, labels)
  
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == labels).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            pbar.set_description('TRAIN loss:%.6f acc:%.6f is_recon=%d' %
                                 (epoch_loss / batch, epoch__acc / batch,is_recon))

        train_loss = running_loss / num_data
        train_acc = running__acc / num_data

        # --- valid ---
        running_loss, running__acc = 0.0, 0.0
        num_data = 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_loader)
            for images, labels in pbar:
                images: Tensor = images.to(DEVICE)
                labels: Tensor = labels.to(DEVICE)
                batch = images.size(0)
                num_data += batch

                output: Tensor = model(images)
                _, pred = torch.max(output, 1)
                loss: Tensor = loss_function(output, labels)

                epoch_loss = loss.item()
                epoch__acc = torch.sum(pred == labels).item()
                running_loss += epoch_loss
                running__acc += epoch__acc

                pbar.set_description('VALID loss:%.6f acc:%.6f' % (
                    epoch_loss / batch, epoch__acc / batch))
                # pbar.set_description('acc:%.6f' % (epoch__acc / batch))
            valid_loss = running_loss / num_data
            valid_acc = running__acc / num_data

        print('Train Loss:%f Accuracy:%f' % (train_loss, train_acc))
        # print('Valid Accuracy:%f' % (valid_acc))
        print('Valid Loss:%f Accuracy:%f' % (valid_loss, valid_acc))

        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/Accuracy', train_acc, global_step=epoch)
        writer.add_scalar('Valid/Loss', valid_loss, global_step=epoch)
        writer.add_scalar('Valid/Accuracy', valid_acc, global_step=epoch)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(
            MODEL_SAVE_DIR, '%s-robust.pt' % MODEL_SAVE_NAME))

        train_log.append([
            epoch,
            train_loss, train_acc,
            valid_loss, valid_acc
        ])

