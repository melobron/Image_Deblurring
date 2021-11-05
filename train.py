import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

import random
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

from dataloaders import GOPRO_Large
from models import RCAN, RNAN, HAN

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Random Seed
seed = random.randint(1, 10000)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Image Deblurring')

# Training parameters
parser.add_argument("--model_name", type=str, default='RNAN_GOPRO_blur_50epochs', help="model_name")
parser.add_argument("--batchSize", type=int, default=5, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--step", type=int, default=200, help="Learning Rate Scheduler Step")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")

# Model specifications
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--reduction', type=float, default=16,
                    help='reduction ratio')

# HAN, RCAN specifications
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')

# RNAN specifications
parser.add_argument('--RAB_num', type=int, default=8,
                    help='number of residual groups')
parser.add_argument('--RNAN_version', type=int, default=1,
                    help='RNAN version')

args = parser.parse_args()

# Deblurring Public Dataset
GOPRO_train_dataset = GOPRO_Large.GOPRO_Large_train_256()
GOPRO_test_dataset = GOPRO_Large.GOPRO_Large_test()

dataset = GOPRO_train_dataset

train_validation_dataset, test_dataset = train_test_split(dataset, train_size=1500, shuffle=False)
train_dataset, validation_dataset = train_test_split(train_validation_dataset, train_size=1200, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batchSize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)

# Model
model = RNAN.RNAN(args=args)
model.to(device)

# Optimizer
criterion = nn.L1Loss(reduction='mean').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-08)


# Learning rate scheduling
def adjust_learning_rate(epoch):
    lr = args.lr * (0.1 ** (epoch // args.step))
    return lr


# PSNR evaluation
def PSNR(pred, gt):
    mse = np.mean((pred / 255. - gt / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# Visualization Functions
def tensor_to_numpy(tensor):
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img


# Checkpoint directory
checkpoint_path = './checkpoints/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


train_loss_list = []
validation_loss_list = []

sharp_tensor, blur_tensor = dataset[0][0].to(device), dataset[0][1].to(device)
sharp_image = tensor_to_numpy(sharp_tensor.cpu())
psnr_list = []


def train(train_dataloader, validation_dataloader, model, optimizer, criterion, epoch):
    # Learning rate scheduler
    lr = adjust_learning_rate(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Training
    model.train()

    for batch, (sharp, blur) in enumerate(train_dataloader):
        sharp, blur = sharp.to(device), blur.to(device)
        loss = criterion(model(blur), sharp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * args.batchSize, len(train_dataloader.dataset),
                       100. * batch / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())

    # Validation
    val_loss = 0
    model.eval()

    for batch, (sharp, blur) in enumerate(validation_dataloader):
        sharp, blur = sharp.to(device), blur.to(device)
        loss = criterion(model(blur), sharp)
        val_loss += loss.item()

        if batch % 10 == 0:
            validation_loss_list.append(loss.item())

    print('Validation Epoch: {} Loss: {:.6f}'.format(
        epoch, val_loss / len(validation_dataloader)))

    # PSNR save
    deblurred_tensor = model(blur_tensor.unsqueeze(0))
    deblurred_image = tensor_to_numpy(deblurred_tensor.squeeze(0).cpu())
    psnr_list.append(PSNR(sharp_image, deblurred_image))

    # checkpoint
    if epoch % 25 == 0:
        checkpoint = {'model': model,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, os.path.join(checkpoint_path, args.model_name + '_' + str(epoch) + '.pth'))


# Training
start = time.time()

for epoch in range(1, args.nEpochs + 1):
    train(train_dataloader, validation_dataloader, model, optimizer, criterion, epoch)

# Save Model
torch.save(model.state_dict(), os.path.join('./trained_models/', args.model_name))

# Save Time
print("time :", time.time() - start)
history = open('runtime_history.txt', 'a')
history.write('model: {}, epoch: {}, batch: {}, runtime: {} \n'.format(args.model_name, args.nEpochs, args.batchSize, time.time() - start))

# Visualize loss
df_train = pd.DataFrame(train_loss_list)
df_validation = pd.DataFrame(validation_loss_list)
df_psnr = pd.DataFrame(psnr_list)

fig, axs = plt.subplots(3)
axs[0].plot(df_train)
axs[0].set_title('train loss')

axs[1].plot(df_validation)
axs[1].set_title('validation loss')

axs[2].plot(df_psnr)
axs[2].set_title('PSNR per epoch')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'graphs', args.model_name + '_loss graph.png'))
plt.show()
