import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
# from abc import abstractmethod
import torch
from PIL import ImageEnhance


"""
Data augmentation scheme.
aug is True/False acc. in get_composed_transform function.
"""
class TransformLoader:
    def __init__(self,
                 image_size, 
                 normalize_param=dict(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4) # not using sharpness
                 ):

        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        """return a transform object

        Args:
            transform_type (str): name of the data augmentation transformation

        Returns:
            the tranform object
        """        
        if transform_type=='ImageJitter':
            method = ImageJitter(transformdict=self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size), int(self.image_size)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param)
        else:
            return method() # these methods go not have arguments

    def get_composed_transform(self, dataset_name, aug=False):
        """Generate a composed transform for dataset_name

        Args:
            dataset_name (str): name of the dataset (determines what type of image transformation to be used)
            aug (bool, optional): whether to use data augmentation. Defaults to False.

        Returns:
            [type]: [description]
        """        
        if  'cifar' in dataset_name.lower() or  'fc100' in dataset_name.lower():
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            if aug:
                print("Using cifar/fc100 specific augmentation strategy")
                transform = transforms.Compose([
                    transforms.RandomCrop(size=32, padding=4), # border is padded with 4 px on each side
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # [max(0, 1 - brightness), 1 + brightness] 
                    transforms.RandomHorizontalFlip(p=0.5),
                    lambda x: np.array(x), # TODO: is this necessary?
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                transform = transforms.Compose([
                    lambda x: np.array(x),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            if aug:
                print("Using MI specific augmentation strategy")
                transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['Resize', 'ToTensor', 'Normalize']
            transform_funcs = [self.parse_transform(x) for x in transform_list]
            transform = transforms.Compose(transform_funcs)


        return transform


"""
Jitter transform: Brightness, Contrast, Color, Sharpness

Copyright 2017-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
transformtypedict=dict(
    Brightness=ImageEnhance.Brightness, 
    Contrast=ImageEnhance.Contrast, 
    Sharpness=ImageEnhance.Sharpness, 
    Color=ImageEnhance.Color
    )

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms)) # random numbers from a uniform distribution

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out