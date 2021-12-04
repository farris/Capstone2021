'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
from skimage import io

class BrainS18Dataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            label_name = os.path.join(self.root_dir, ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = io.imread(img_name)
            assert img is not None
            mask = np.load(label_name)
            mask = np.mean(mask.f.arr_0)
            assert mask is not None
            
            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            #mask_array = self.__nii2tensorarray__(mask_array)

            #assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            
            return img_array, mask_array
        
        elif self.phase == "test":
            # read image
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(self.root_dir, ith_info[0])
            assert os.path.isfile(img_name)
            img = io.imread(img_name)
            assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)

            return img_array

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data, label = self.__random_center_crop__ (data, label)
        
        return data, label

    def __training_data_process__(self, data, label): 
        # crop data according net input size
        
        # drop out the invalid range
        #data, label = self.__drop_invalid_range__(data, label)
        
        # crop data
        #data, label = self.__crop_data__(data, label) 

        # resize data
        data = self.__resize_data__(data)
        #label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data): 
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data