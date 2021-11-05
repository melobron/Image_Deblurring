import torch
import torchvision.transforms as transforms

from dataloaders import GOPRO_Large
from models import RCAN, RNAN, HAN, MIMOUNet

import math
import numpy as np
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt
import argparse

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

parser = argparse.ArgumentParser(description='Super Resolution')

# Training parameters
parser.add_argument("--model_name", type=str, default='?', help="model_name")

# Model specifications
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--reduction', type=float, default=16,
                    help='reduction ratio')
parser.add_argument('--upsample_ratio', type=float, default=4,
                    help='upsample ratio')

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
GOPRO_train_dataset = GOPRO_Large.GOPRO_Large_train_64(blur_gamma=False)
GOPRO_test_dataset = GOPRO_Large.GOPRO_Large_test(blur_gamma=False)

train_dataset = GOPRO_train_dataset
test_dataset = GOPRO_test_dataset
print("{} images in the train dataset".format(len(train_dataset)))
print("{} images in the test dataset".format(len(test_dataset)))

# Model1
model = MIMOUNet.MIMOUNet()
# model.load_state_dict(torch.load('./checkpoints/?.pth', map_location='cuda')['state_dict'])
model.load_state_dict(torch.load('./trained_models/MIMO_GOPRO_blur_500epochs', map_location='cuda'))
model.to(device)
model.eval()


# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("model param num: {}".format(count_parameters(model)))


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


##################################################################################################################
# Average PSNR
psnr = 0

for i in range(0, len(test_dataset)):
    sharp_tensor = test_dataset[i][0]
    blur_tensor = test_dataset[i][1].to(device)
    # deblurred_tensor = model(blur_tensor.unsqueeze(0))[2].squeeze(0)  # only for MIMOUNet
    deblurred_tensor = model(blur_tensor.unsqueeze(0)).squeeze(0)

    sharp_image = tensor_to_numpy(sharp_tensor)
    deblurred_image = tensor_to_numpy(deblurred_tensor.cpu())

    psnr += PSNR(sharp_image, deblurred_image)
    print("{}th images calculated".format(i+1))

print('total PSNR: {}'.format(psnr/len(test_dataset)))

##################################################################################################################
# Average SSIM

