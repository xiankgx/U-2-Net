# data loader
from __future__ import division, print_function

import os
import glob
import random

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from skimage import color, io, transform
from torch.utils.data import Dataset
from torchvision import transforms


def get_heavy_transform(transform_size=True, width=288, height=288):
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Rotate(limit=45, p=0.5,
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT),

        A.OneOf([
            A.GaussianBlur(),
            A.MedianBlur(),
            A.MotionBlur(),
        ], p=0.2),

        A.OneOf([
            A.IAAPiecewiseAffine(),
            A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT)
        ], p=0.2),

        # A.RandomSunFlare(p=0.1),
        # A.RandomFog(p=0.1),

        A.OneOf([
            A.CLAHE(),
            A.RandomGamma(),
            A.HueSaturationValue(),
            A.ToSepia(),
            A.RGBShift(),
            A.ChannelShuffle(),
        ], p=0.2),

        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2,
                                   p=0.5),
        A.ToGray(p=0.15),

        A.OneOf([
            A.GaussNoise(var_limit=(0, 25)),
            A.ISONoise()
        ], p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.99,
                    p=0.2),
        A.JpegCompression(quality_lower=65, quality_upper=100,
                          p=0.2)
    ]
        + ([A.RandomResizedCrop(height=height, width=width,
                                scale=(0.5, 1.5),
                                ratio=(0.5, 2.0),
                                interpolation=cv2.INTER_LINEAR), ] if transform_size else [])
    )

# ==========================dataset load==========================


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            # Resize shortest edge to size
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size),
                               mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size),
                               mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w),
                               mode='constant')
        lbl = transform.resize(label, (new_h, new_w),
                               mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        image = image/np.max(image)
        if(np.max(label) < 1e-6):
            label = label
        else:
            label = label/np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485)/0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485)/0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485)/0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485)/0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456)/0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406)/0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if(np.max(label) < 1e-6):
            label = label
        else:
            label = label/np.max(label)

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0]-np.min(tmpImgt[:, :, 0])) / \
                (np.max(tmpImgt[:, :, 0])-np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1]-np.min(tmpImgt[:, :, 1])) / \
                (np.max(tmpImgt[:, :, 1])-np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2]-np.min(tmpImgt[:, :, 2])) / \
                (np.max(tmpImgt[:, :, 2])-np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0]-np.min(tmpImgtl[:, :, 0])) / \
                (np.max(tmpImgtl[:, :, 0])-np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1]-np.min(tmpImgtl[:, :, 1])) / \
                (np.max(tmpImgtl[:, :, 1])-np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2]-np.min(tmpImgtl[:, :, 2])) / \
                (np.max(tmpImgtl[:, :, 2])-np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (
                tmpImg[:, :, 0]-np.mean(tmpImg[:, :, 0]))/np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (
                tmpImg[:, :, 1]-np.mean(tmpImg[:, :, 1]))/np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (
                tmpImg[:, :, 2]-np.mean(tmpImg[:, :, 2]))/np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (
                tmpImg[:, :, 3]-np.mean(tmpImg[:, :, 3]))/np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (
                tmpImg[:, :, 4]-np.mean(tmpImg[:, :, 4]))/np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (
                tmpImg[:, :, 5]-np.mean(tmpImg[:, :, 5]))/np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0]-np.min(tmpImg[:, :, 0])) / \
                (np.max(tmpImg[:, :, 0])-np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1]-np.min(tmpImg[:, :, 1])) / \
                (np.max(tmpImg[:, :, 1])-np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2]-np.min(tmpImg[:, :, 2])) / \
                (np.max(tmpImg[:, :, 2])-np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (
                tmpImg[:, :, 0]-np.mean(tmpImg[:, :, 0]))/np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (
                tmpImg[:, :, 1]-np.mean(tmpImg[:, :, 1]))/np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (
                tmpImg[:, :, 2]-np.mean(tmpImg[:, :, 2]))/np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image/np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485)/0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485)/0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485)/0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485)/0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456)/0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406)/0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': imidx, 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        # imidx = np.array([idx])
        imidx = os.path.dirname(imname).replace("/", "__").replace("..__data__", "") + "__" + os.path.splitext(os.path.basename(imname))[0]

        if image.ndim == 2:
            image = np.stack([image, ] * 3, axis=-1)

        if(0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        # Make equal spatial dimension
        if label_3.shape[:2] != image.shape[:2]:
            image = transform.resize(image, label_3.shape[:2])

        label = np.zeros(label_3.shape[0:2])
        if(3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif(2 == len(label_3.shape)):
            label = label_3

        if(3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif(2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class AlbuSampleTransformer(object):
    """Meta transformer for applying albumentations transform to a sample."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        try:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"].astype(np.float32)
            label = transformed["mask"].astype(np.float32)
        except Exception as e:
            print(
                f"{imidx} -> {str(e)}, image shape: {image.shape}, label shape: {label.shape}")

        return {'imidx': imidx, 'image': image, 'label': label}


class SaveDebugSamples(object):

    def __init__(self, out_dir="./debug/", p_sample=0.03):
        self.out_dir = out_dir
        self.p_sample = p_sample

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if np.random.rand() < self.p_sample:
            # print(f"image - shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
            # print(f"label - shape: {label.shape}, dtype: {label.dtype}, min: {label.min()}, max: {label.max()}")

            os.makedirs(self.out_dir, exist_ok=True)

            image_path = os.path.join(self.out_dir, f"{imidx}_image.png")
            Image.fromarray(image.astype(np.uint8)).save(image_path)

            label_path = os.path.join(self.out_dir, f"{imidx}_label.png")
            Image.fromarray(label[..., 0].astype(np.uint8)).save(label_path)

        return sample


class MultiScaleSalObjDataset(SalObjDataset):
    """Salient object detection dataset for multi-scale training."""

    def __init__(self,
                 *pargs,
                 sizes=[256, 320, 384, 448, 512],
                 **kwargs):
        super(MultiScaleSalObjDataset, self).__init__(*pargs, **kwargs)

        self.sizes = sizes
        self.transform_size_list = [
            transforms.Compose([
                AlbuSampleTransformer(A.RandomResizedCrop(width=size, height=size,
                                                          scale=(0.5, 1.5),
                                                          ratio=(0.5, 2.0),
                                                          interpolation=cv2.INTER_LINEAR)),
                ToTensorLab(flag=0)
            ])
            for size in sizes
        ]

    def __getitem__(self, idx):
        sample = super(MultiScaleSalObjDataset, self).__getitem__(idx)

        ms_sample = {}
        ms_sample["imidx"] = sample["imidx"]
        for i, size in enumerate(self.sizes):
            _sample = self.transform_size_list[i](sample)
            ms_sample[f"image_{size}"] = _sample["image"]
            ms_sample[f"label_{size}"] = _sample["label"]

        return ms_sample


class MixupAugSalObjDataset(SalObjDataset):
    """Saliency object detection dataset with mixup data augmentation."""

    def __init__(self, alpha=0.2, *pargs, **kwargs):
        super(MixupAugSalObjDataset, self).__init__(*pargs, **kwargs)
        self.alpha = alpha

    def __getitem__(self, idx):
        sample_1 = super(MixupAugSalObjDataset, self).__getitem__(idx)

        idx_2 = np.random.randint(len(self))
        sample_2 = super(MixupAugSalObjDataset, self).__getitem__(idx_2)

        lam = np.random.beta(self.alpha, self.alpha)

        sample = {
            'imidx_1': sample_1['imidx'],
            'imidx_2': sample_2['imidx'],
            'image': lam * sample_1['image'] + (1.0 - lam) * sample_2['image'],
            'label': lam * sample_1['label'] + (1.0 - lam) * sample_2['label']
        }
        return sample
