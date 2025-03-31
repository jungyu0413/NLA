# -*- coding: utf-8 -*-
import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from src.utils import *
from torch.utils.data import Dataset



class NLA_Rafdb(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.noise = args.noise
        self.imbalanced = args.imbalanced
        self.test_label_path = args.label_path
        self.dataset_path = args.dataset_path
        if self.noise:
            self.noise_name = args.noise_name
            self.label_path = args.label_path.split('.')[0] + '_' + self.noise_name + '.txt'
        elif self.imbalanced:
            self.imbalanced_name = args.imbalanced_name
            self.label_path = args.imbalanced_path.split('.')[0] + '_' + self.imbalanced_name + '.txt'
        else:
            self.label_path = args.label_path
            
        
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        
        df = pd.read_csv(self.label_path, sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
            if self.noise:
                self.rn_check = df[2]
            
        else:
            df = pd.read_csv(self.test_label_path, sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = True
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.dataset_path, 'Image/aligned', f)
            self.file_paths.append(file_name)


    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        label = self.label
        return label
    
    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]
        
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        flip_image = transforms.RandomHorizontalFlip(p=1)(image)
        

        return image, label, flip_image

    
    

class NLA_Affecnet(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.noise = args.noise
        self.imbalanced = args.imbalanced
        self.imbalanced_name = args.imbalanced_name
        if phase == 'train':
            if self.noise:
                self.noise_name = args.noise_name
                self.label_path = args.noise_path.split('.')[0] + '_' + self.noise_name + '.csv'
                self.test_label_path = args.label_path
                
            elif self.imbalanced:
                self.label_path = args.imbalanced_path.split('.')[0] + '_' + self.imbalanced_name + '.csv'
                self.test_label_path = args.label_path
            else:
                self.label_path = os.path.join(args.dataset_path, 'align_crop_train.csv')
        else:
            self.label_path = os.path.join(args.dataset_path, 'align_crop_test.csv')

        
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        
        self.df = pd.read_csv(self.label_path)

        self.dataset = self.df[self.df.expression.isin(list(range(args.num_classes)))]
        self.label = list(self.dataset.expression)
        self.file_paths = list(self.dataset.path)
        
        self.aug_func = [flip_image, add_g]
        self.image_paths = self.dataset['path']
        self.clean = True



    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        label = self.label
        return label
    
    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        image = cv2.imread(self.image_paths[idx])
        image = image[:, :, ::-1]
    
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        flip_image = transforms.RandomHorizontalFlip(p=1)(image)
        
        return image, label, flip_image

    

    
class NLA_ferplus(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.noise = args.noise
        self.imbalanced = args.imbalanced
        self.imbalanced_name = args.imbalanced_name
        if self.noise:
            self.noise_name = args.noise_name
            self.label_path = args.dataset_path.split('.')[0] + '_' + self.noise_name + '.csv'
            self.test_label_path = args.label_path
            
        elif self.imbalanced:
            self.label_path = args.imbalanced_path.split('.')[0] + '_' + self.imbalanced_name + '.csv'
            self.test_label_path = args.label_path
        else:
            self.label_path = args.label_path
            self.test_label_path = args.label_path
        
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        
        self.df = pd.read_csv(self.label_path)
        self.test_df = pd.read_csv(self.test_label_path)
        
        
        if phase == 'Train':
            self.dataset = self.df[self.df['type'] == phase].reset_index()
            self.label = self.dataset["label"]
            if self.noise:
                self.rn_check = self.dataset['noise']
        else:   
            self.dataset = self.test_df[self.test_df['type'] == phase].reset_index()
            self.label = self.test_df["label"]

        self.image_paths = self.dataset["path"]
        
        self.aug_func = [flip_image, add_g]
        self.file_paths = self.dataset['path']
        self.clean = True
    

    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        label = self.label
        return label
    
    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        image = cv2.imread(self.image_paths[idx])
        image = image[:, :, ::-1]
        
        if self.phase == 'Train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        flip_image = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, flip_image
    
    
    
    
    









class Integrated_Affecnet(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.noise = args.noise
        self.imbalanced = args.imbalanced
        self.imbalanced_name = args.imbalanced_name
        if phase == 'train':
            if self.noise:
                self.noise_name = args.noise_name
                self.label_path = args.noise_path.split('.')[0] + '_' + self.noise_name + '.csv'
                self.test_label_path = args.label_path
                
            elif self.imbalanced:
                self.label_path = args.imbalanced_path.split('.')[0] + '_' + self.imbalanced_name + '.csv'
                self.test_label_path = args.label_path
            else:
                self.label_path = os.path.join(args.dataset_path, 'align_crop_train.csv')
        else:
            self.label_path = os.path.join(args.dataset_path, 'align_crop_test.csv')

        
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        
        self.df = pd.read_csv(self.label_path)

        self.dataset = self.df[self.df.expression.isin(list(range(args.num_classes)))]
        self.label = list(self.dataset.expression)
        self.file_paths = list(self.dataset.path)
        if self.noise and phase == 'train':
            self.rn_check = self.dataset['noise']
        
        self.aug_func = [flip_image, add_g]
        self.image_paths = self.dataset['path']
        self.clean = True
        



    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        label = self.label
        return label
    
    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        image = cv2.imread(self.image_paths[idx])
        image = image[:, :, ::-1]
        path = self.file_paths[idx]
        
    
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        image1 = transforms.RandomHorizontalFlip(p=1)(image)
        
        if self.noise:
            if self.phase == 'train':
                rn_check = self.rn_check[idx]
            else:
                rn_check ='None'
        else:
            rn_check ='None'
        
        if self.noise:
            return image, label, image1, path, rn_check
        else:
            return image, label, image1, path, rn_check

