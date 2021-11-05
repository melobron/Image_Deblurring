import torch
import torchvision.transforms as transforms

from glob import glob
import os
import cv2


def load_img(image_dir):
    image_path = os.path.join(os.getcwd(), image_dir)
    image_list = glob((os.path.join(image_path, '*.png')))
    return image_list


def get_img(image_path):
    image_numpy = cv2.imread(image_path)
    return image_numpy


def get_patches(image_numpy, patch_size, interval):
    image_patches = []
    image_h, image_w, _ = image_numpy.shape

    for h in range(patch_size, image_h, interval):
        for w in range(patch_size, image_w, interval):
            patch = image_numpy[h-patch_size:h, w-patch_size:w, :]
            image_patches.append(patch)

    return image_patches


def data_augmentation(image_dir, patch_size, save_path='sharp'):
    image_list = load_img(image_dir)

    for index in range(len(image_list)):
        image_path = image_list[index]
        gopro_path = os.path.dirname(os.path.dirname(image_path))
        train_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))

        gopro_dir = os.path.basename(gopro_path)  # GOPR0372_07_00
        train128_path = os.path.join(train_path, 'test_64')  # \Users\User\Desktop\Image_Deblurring\datasets/GOPRO_Large\train_256
        new_gopro_dir = os.path.join(train128_path, gopro_dir)  # \Users\User\Desktop\Image_Deblurring\datasets/GOPRO_Large\train_256\GOPR0372_07_00s

        os.makedirs(os.path.join(new_gopro_dir, 'blur'), exist_ok=True)
        os.makedirs(os.path.join(new_gopro_dir, 'blur_gamma'), exist_ok=True)
        os.makedirs(os.path.join(new_gopro_dir, 'sharp'), exist_ok=True)

        image_filename = os.path.basename(image_path)
        image_filename = os.path.splitext(image_filename)[0]
        image_numpy = get_img(image_path)

        image_patches = get_patches(image_numpy, patch_size, interval=patch_size*20)
        for patch_index in range(len(image_patches)):
            image_name = image_filename + '_' + str(patch_index) + '.png'
            image_patch = image_patches[patch_index]
            cv2.imwrite(os.path.join(new_gopro_dir, save_path, image_name), image_patch)
        print('data no. {} saved'.format(index+1))
    print('all saved')


sharp_path = 'datasets/GOPRO_Large/test/**/sharp/'
blur_path = 'datasets/GOPRO_Large/test/**/blur/'
blur_gamma_path = 'datasets/GOPRO_Large/test/**/blur_gamma/'
patch_size = 64

data_augmentation(sharp_path, patch_size, 'sharp')
data_augmentation(blur_path, patch_size, 'blur')
data_augmentation(blur_gamma_path, patch_size, 'blur_gamma')