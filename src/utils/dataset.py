import os
import cv2
import torch
import numpy as np
import albumentations as albu

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, dir: str, img_size: int, augmentation: bool):
        self.dir = dir
        self.img_size = img_size
        self.augmentation = augmentation
        
    def get_training_augmentation(self):
        train_transform = [
            # albu.GridDistortion(p=0.3),
            albu.Transpose(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.VerticalFlip(p=0.4),
            # albu.Resize(height=self.img_size, width=self.img_size, interpolation=1, always_apply=False, p=1),
            albu.RandomCrop(width=self.img_size, height=self.img_size),
            albu.RandomRotate90(p=0.4),
            albu.Flip(p=0.4),
            albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.1)
            # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        ]
        return albu.Compose(train_transform)
    
    def load_folder(self):
        name_images = [item for item in os.listdir(f'{self.dir}/') if item[-8:-4] != "mask"]
        name_masks = [item for item in os.listdir(f'{self.dir}/') if item[-8:-4] == "mask"]
        clear_image, clear_mask = [], []
        for name_image, name_mask in zip(name_images, name_masks):
            image = cv2.imread(f'{self.dir}/{name_image}')
            mask = cv2.imread(f'{self.dir}/{name_mask}')*255
            clear_image.append(image)
            clear_mask.append(mask)
        return clear_image, clear_mask
    
    def prepocessing_image(self, image):
        image = np.array(Image.fromarray(image).resize((self.img_size, self.img_size))).astype('float32')/255
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = (np.rint(img_gray)).astype(int)
        return img_gray
    
    def preprocessing_data(self, data, images=True):
        array = []
        if images == True:
            for image in data:
                image = (cv2.resize(image, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)).astype('float32')/255
                # image = (image - np.mean(image))/np.std(image)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_gray = (np.rint(img_gray)).astype(np.uint8)
                array.append(img_gray)
        else:
            for image in data:
                mask = (cv2.resize(image, (self.img_size, self.img_size), interpolation = cv2.INTER_AREA)).astype('float32')
                mask = mask.max(axis=2)/255
                array.append(mask)
        return array
    
    def Generator(self, batch_size, X_train, X_test, y_train, y_test):
        X_train_prep = self.preprocessing_data(X_train, images=True)
        X_test_prep = self.preprocessing_data(X_test, images=True)
        y_train_prep = self.preprocessing_data(y_train, images=False)
        y_test_prep = self.preprocessing_data(y_test, images=False)
        if self.augmentation == False:
            train_dataset = Dataset(X_train_prep, y_train_prep)
        else:
            train_dataset = Dataset(X_train_prep, y_train_prep, augmentation=self.get_training_augmentation())
        valid_dataset = Dataset(X_test_prep, y_test_prep, False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, valid_loader

class Dataset(BaseDataset):
    def __init__(
            self, 
            images, 
            masks, 
            augmentation=None
    ):
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
    
    def __getitem__(self, i):
        image = self.images[i]
        mask = self.masks[i]
        flag = np.random.rand() > 0.3
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            while mask.sum() == 0 and flag:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
        
        
        return {
            "image" : torch.tensor(np.array([image]), dtype = torch.float),
            "mask" : torch.tensor(np.array([mask]), dtype = torch.float)
            }
        
    def __len__(self):
        return len(self.images)