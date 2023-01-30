import os
import cv2
import torch
import numpy as np
import albumentations as albu

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from sklearn.model_selection import train_test_split

def load_data(test_size: int, batch_size: int, img_size: int, dir: str, artificial_increase: int):
    
    images, masks = preprocessing_folder(img_size, dir)
    
    images*=artificial_increase
    masks*=artificial_increase
    
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=42)
    
    train_dataset = Dataset(X_train, y_train, augmentation=get_training_augmentation(img_size))
    valid_dataset = Dataset(X_test, y_test, False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader

def get_training_augmentation(IMG_SIZE):
    
    train_transform = [
        
        albu.RandomCrop(width = IMG_SIZE, height = IMG_SIZE),
        albu.RandomRotate90(),
        albu.Flip(p=0.5),
#         albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)

def preprocessing_folder(IMG_SIZE, dir: str):
    
    get_folder_dir = os.listdir(dir)
    
    name_images = os.listdir(f'{dir}/{get_folder_dir[0]}/')
    name_masks = os.listdir(f'{dir}/{get_folder_dir[1]}/')
    
    array_images, array_masks = [], []
    
    for name_image, name_mask in zip(name_images, name_masks):

        image = cv2.imread(f'{dir}/{get_folder_dir[0]}/{name_image}')
        image = np.array(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))).astype('float32')/255
        
        img = (image - np.mean(image))/np.std(image)

        img_gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        img_gray = np.moveaxis(img_gray, -1, 0)
        
        mask = cv2.imread(f'{dir}/{get_folder_dir[1]}/{name_mask}')*255
        mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE))).astype('float32')
        mask = mask.max(axis=2)/255

        array_images.append(img_gray)
        array_masks.append(mask)
    
    return array_images, array_masks

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