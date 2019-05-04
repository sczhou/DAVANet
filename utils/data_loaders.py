#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import cv2
import json
import numpy as np
import os
import io
import random
import scipy.io
import sys
import torch.utils.data.dataset

from config import cfg
from datetime import datetime as dt
from enum import Enum, unique
from utils.imgio_gen import readgen
import utils.network_utils

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1

class FlyingThings3DDataset(torch.utils.data.dataset.Dataset):
    """DrivingDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms = None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        imgs, disps = self.get_datum(idx)
        imgs, disps = self.transforms(imgs, disps)
        if cfg.DATASET.WITH_MASK:
            occs = utils.network_utils.get_occ([img.view(1, *img.shape) for img in imgs], [disp * cfg.DATA.DIV_DISP for disp in disps], cuda=False)
        else:
            _, H, W = imgs[0].shape
            occs = [torch.ones((1,H,W), dtype=torch.float32), torch.ones((1,H,W), dtype=torch.float32)]
        name = []
        return name, imgs, disps, occs

    def get_datum(self, idx):
        img_left_path    = self.file_list[idx]['img_left']
        img_right_path   = self.file_list[idx]['img_right']
        disp_left_path   = self.file_list[idx]['disp_left']
        disp_right_path  = self.file_list[idx]['disp_right']

        img_left  = readgen(img_left_path).astype(np.float32)
        img_right = readgen(img_right_path).astype(np.float32)
        imgs = [img_left, img_right]

        disp_left = readgen(disp_left_path).astype(np.float32)
        disp_right = readgen(disp_right_path).astype(np.float32)

        disps = [disp_left, disp_right]
        return imgs, disps
# //////////////////////////////// = End of FlyingThings3DDataset Class Definition = ///////////////////////////////// #

class FlyingThings3DDataLoader:
    def __init__(self):
        self.img_left_path_template = cfg.DIR.IMAGE_LEFT_PATH
        self.img_right_path_template = cfg.DIR.IMAGE_RIGHT_PATH
        self.disp_left_path_template = cfg.DIR.DISPARITY_LEFT_PATH
        self.disp_right_path_template = cfg.DIR.DISPARITY_RIGHT_PATH
        # Load all files of the dataset
        with io.open(cfg.DIR.DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())

    def get_dataset(self, dataset_type, transforms=None):
        files = []
        # Load data for each category
        for file in self.files_list:
            if dataset_type == DatasetType.TRAIN and (file['phase'] == 'TRAIN' or file['phase'] == 'TEST'):
                categories = file['categories']
                phase = file['phase']
                classes = file['classes']
                names = file['names']
                samples = file['sample']
                print('[INFO] %s Collecting files of Taxonomy [categories = %s, phase = %s, classes = %s, names = %s]' % (
                dt.now(), categories, phase, classes, names))
                files.extend(
                    self.get_files_of_taxonomy(categories, phase, classes, names, samples))
            elif dataset_type == DatasetType.TEST and file['phase'] == 'TEST':
                categories = file['categories']
                phase = file['phase']
                classes = file['classes']
                names = file['names']
                samples = file['sample']
                print('[INFO] %s Collecting files of Taxonomy [categories = %s, phase = %s, classes = %s, names = %s]' % (
                    dt.now(), categories, phase, classes, names))
                files.extend(
                    self.get_files_of_taxonomy(categories, phase, classes, names, samples))

        print('[INFO] %s Complete collecting files of the dataset for %s. Total files: %d.' % (dt.now(), dataset_type.name, len(files)))
        return FlyingThings3DDataset(files, transforms)

    def get_files_of_taxonomy(self,categories, phase, classes, names, samples):

        # n_samples = len(samples)
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_left_path = self.img_left_path_template % (categories, phase, classes, names, sample_name)
            img_right_path = self.img_right_path_template % (categories, phase, classes, names, sample_name)
            disp_left_path = self.disp_left_path_template % (phase, classes, names, sample_name)
            disp_right_path = self.disp_right_path_template % (phase, classes, names, sample_name)

            if os.path.exists(img_left_path) and os.path.exists(img_right_path) and os.path.exists(
                    disp_left_path) and os.path.exists(disp_right_path):
                files_of_taxonomy.append({
                    'img_left': img_left_path,
                    'img_right': img_right_path,
                    'disp_left': disp_left_path,
                    'disp_right': disp_right_path,
                    'categories': categories,
                    'classes' : classes,
                    'names': names,
                    'sample_name': sample_name
                })
        return files_of_taxonomy
# /////////////////////////////// = End of FlyingThings3DDataLoader Class Definition = /////////////////////////////// #

class StereoDeblurDataset(torch.utils.data.dataset.Dataset):
    """StereoDeblurDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms = None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, imgs, disps = self.get_datum(idx)
        imgs, disps = self.transforms(imgs, disps)
        occs = utils.network_utils.get_occ([img.view(1, *img.shape) for img in imgs[-2:]], [disp * cfg.DATA.DIV_DISP for disp in disps], cuda=False)
        # remove nan and inf pixel
        disps[0][occs[0]==0] = 0
        disps[1][occs[1]==0] = 0
        return name, imgs, disps, occs

    def get_datum(self, idx):

        name = self.file_list[idx]['name']
        img_blur_left_path = self.file_list[idx]['img_blur_left']
        img_blur_right_path = self.file_list[idx]['img_blur_right']
        img_clear_left_path = self.file_list[idx]['img_clear_left']
        img_clear_right_path = self.file_list[idx]['img_clear_right']
        disp_left_path = self.file_list[idx]['disp_left']
        disp_right_path = self.file_list[idx]['disp_right']

        img_blur_left = readgen(img_blur_left_path).astype(np.float32)
        img_blur_right = readgen(img_blur_right_path).astype(np.float32)
        img_clear_left = readgen(img_clear_left_path).astype(np.float32)
        img_clear_right = readgen(img_clear_right_path).astype(np.float32)
        imgs = [img_blur_left, img_blur_right, img_clear_left, img_clear_right]

        disp_left = readgen(disp_left_path).astype(np.float32)
        disp_right = readgen(disp_right_path).astype(np.float32)

        disps = [disp_left, disp_right]
        return name, imgs, disps
# //////////////////////////////// = End of StereoDeblurDataset Class Definition = ///////////////////////////////// #

class StereoDeblurLoader:
    def __init__(self):
        self.img_left_blur_path_template = cfg.DIR.IMAGE_LEFT_BLUR_PATH
        self.img_left_clear_path_template = cfg.DIR.IMAGE_LEFT_CLEAR_PATH
        self.img_right_blur_path_template = cfg.DIR.IMAGE_RIGHT_BLUR_PATH
        self.img_right_clear_path_template = cfg.DIR.IMAGE_RIGHT_CLEAR_PATH
        self.disp_left_path_template = cfg.DIR.DISPARITY_LEFT_PATH
        self.disp_right_path_template = cfg.DIR.DISPARITY_RIGHT_PATH
        # Load all files of the dataset
        with io.open(cfg.DIR.DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())

    def get_dataset(self, dataset_type, transforms=None):
        files = []
        # Load data for each sequence
        for file in self.files_list:
            if dataset_type == DatasetType.TRAIN and file['phase'] == 'Train':
                name = file['name']
                pair_num = file['pair_num']
                samples = file['sample']
                files_num_old = len(files)
                files.extend(self.get_files_of_taxonomy(name, samples))
                print('[INFO] %s Collecting files of Taxonomy [Name = %s, Pair Numbur = %s, Loaded = %r]' % (
                    dt.now(), name, pair_num, pair_num == (len(files)-files_num_old)))
            elif dataset_type == DatasetType.TEST and file['phase'] == 'Test':
                name = file['name']
                pair_num = file['pair_num']
                samples = file['sample']
                files_num_old = len(files)
                files.extend(self.get_files_of_taxonomy(name, samples))
                print('[INFO] %s Collecting files of Taxonomy [Name = %s, Pair Numbur = %s, Loaded = %r]' % (
                    dt.now(), name, pair_num, pair_num == (len(files)-files_num_old)))

        print('[INFO] %s Complete collecting files of the dataset for %s. Total Pair Numbur: %d.\n' % (dt.now(), dataset_type.name, len(files)))
        return StereoDeblurDataset(files, transforms)

    def get_files_of_taxonomy(self, name, samples):

        # n_samples = len(samples)
        files_of_taxonomy = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_left_clear_path = self.img_left_clear_path_template % (name, sample_name)
            img_right_clear_path = self.img_right_clear_path_template % (name, sample_name)
            img_left_blur_path = self.img_left_blur_path_template % (name, sample_name)
            img_right_blur_path = self.img_right_blur_path_template % (name, sample_name)
            disp_left_path = self.disp_left_path_template % (name, sample_name)
            disp_right_path = self.disp_right_path_template % (name, sample_name)

            if os.path.exists(img_left_blur_path) and os.path.exists(img_right_blur_path) and os.path.exists(
                    img_left_clear_path) and os.path.exists(img_right_clear_path) and os.path.exists(
                    disp_left_path) and os.path.exists(disp_right_path):
                files_of_taxonomy.append({
                    'name': name,
                    'img_blur_left': img_left_blur_path,
                    'img_blur_right': img_right_blur_path,
                    'img_clear_left': img_left_clear_path,
                    'img_clear_right': img_right_clear_path,
                    'disp_left': disp_left_path,
                    'disp_right': disp_right_path,
                })
        return files_of_taxonomy
# /////////////////////////////// = End of StereoDeblurLoader Class Definition = /////////////////////////////// #


DATASET_LOADER_MAPPING = {
    'FlyingThings3D': FlyingThings3DDataLoader,
    'StereoDeblur': StereoDeblurLoader,
}
