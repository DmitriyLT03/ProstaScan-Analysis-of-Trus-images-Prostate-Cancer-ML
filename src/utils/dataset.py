import os
import cv2
import torch
import numpy as np
import albumentations as albu

from PIL import Image
from torch.utils.data import Dataset as BaseDataset

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
        
        if self.augmentation:
            sample = self.augmentation(
                image=image,
                mask=mask
            )
            image, mask = sample['image'], sample['mask']
        
        return {
            "image" : torch.tensor(np.array(image), dtype = torch.float),
            "mask" : torch.tensor(np.array([mask]), dtype = torch.float)
            }
        
    def __len__(self):
        return len(self.images)

class Preprocessing:
    def __init__(self, dir: str, img_size: int, augmentation: bool):
        self.dir = dir
        self.img_size = img_size
        self.augmentation = augmentation
        
    def Generator(self, batch_size, X_train, X_test, y_train, y_test):
        
        X_train_prep = self.preprocessing_data(
            X_train,
            images=True
        )
        X_test_prep = self.preprocessing_data(
            X_test, 
            images=True
        )
        y_train_prep = self.preprocessing_data(
            y_train, 
            images=False
        )
        y_test_prep = self.preprocessing_data(
            y_test, 
            images=False
        )
        
        if self.augmentation == False:
            train_dataset = Dataset(
                X_train_prep,
                y_train_prep
            )
        else:
            train_dataset = Dataset(
                X_train_prep,
                y_train_prep,
                augmentation=self.get_training_augmentation()
            )
            
        valid_dataset = Dataset(
            X_test_prep,
            y_test_prep,
            False
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        return train_loader, valid_loader
    
    def load_folder(self):
        name_images = [item for item in os.listdir(f'{self.dir}/') if item[-8:-4] != "mask"]
        name_masks = [item for item in os.listdir(f'{self.dir}/') if item[-8:-4] == "mask"]
        clear_image, clear_mask = [], []
        for name_image, name_mask in zip(name_images, name_masks):
            image = cv2.imread(
                f'{self.dir}/{name_image}'
            )
            mask = cv2.imread(
                f'{self.dir}/{name_mask}'
            )*255
            image = image[135:750, 145:1125]
            mask = mask[135:750, 145:1125]
            clear_image.append(image)
            clear_mask.append(mask)
        return clear_image, clear_mask
        
    def prepocessing_image(self, image):
        # normalizator = self.Normalize()
        # image = normalizator(image=image)['image']
        b, g, r = cv2.split(image)
        image_3d = cv2.merge([b, g, r])
        image_3d = (255 - image_3d)
        resize_image = (cv2.resize(
            image_3d, 
            (self.img_size, self.img_size),
            interpolation = cv2.INTER_AREA)
        ).astype('float32')/255
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)
        image = (np.rint(resize_image)).astype(np.uint8)
        image = np.moveaxis(image, -1, 0)
        return image
    
    def preprocessing_data(self, data, images=True):
        array = []
        if images == True:
            for image in data:
                preproc_img = self.prepocessing_image(image)
                array.append(preproc_img)
        else:
            for image in data:
                mask = (cv2.resize(
                    image, 
                    (self.img_size, self.img_size),
                    interpolation = cv2.INTER_AREA)
                ).astype('float32')
                
                mask = mask.max(axis=2)/255
                array.append(mask)
        return array
    
    def inference_image(self, image):
        image = image[135:750, 145:1125]
        return self.prepocessing_image(image)
    
    def get_training_augmentation(self):
        train_transform = [
            # albu.OneOf([
            #         albu.CropNonEmptyMaskIfExists(
            #             height=256, 
            #             width=256, 
            #             ignore_values=None, 
            #             ignore_channels=None, 
            #             always_apply=False,
            #             p=1.0
            #         ),
            #         albu.PadIfNeeded(
            #             min_height=128,
            #             min_width=128,
            #             p=0.5
            #         )
            #     ], p=1),
            # albu.OneOf([
            #     albu.ElasticTransform(
            #         alpha=120, 
            #         sigma=120 * 0.05, 
            #         alpha_affine=120 * 0.03,
            #         p=0.5
            #     ),
            #     albu.GridDistortion(p=0.5),
            #     albu.OpticalDistortion(
            #         distort_limit=2,
            #         shift_limit=0.5, 
            #         p=1
            #     )                  
            #     ], p=0.8),
            # albu.GridDistortion(p=0.1),
            # albu.Transpose(p=0.4),
            # albu.HorizontalFlip(p=0.15),
            # albu.VerticalFlip(p=0.1),
            # albu.RandomRotate90(p=0.15),
            # albu.Flip(p=0.14),
            # albu.ColorJitter(
            #     brightness=0.5, 
            #     contrast=0.5,
            #     saturation=0.2, 
            #     hue=0.2,
            #     always_apply=False, 
            #     p=0.12
            # ),
            # albu.RandomBrightnessContrast(p=0.13),    
            # albu.RandomGamma(p=0.1)
            # albu.Resize(
            #     height=256,
            #     width=256
            # )]
            # albu.OneOf([
            #     albu.CLAHE(clip_limit=2),
            # ], p=.35),
            # albu.OneOf([
            #     albu.GaussNoise(p=.7),
            # ], p=.5),
            # albu.RandomRotate90(p=0.4),
            # albu.Flip(p=0.4),
            # albu.Transpose(p=0.4),
            # albu.OneOf([
            #     albu.MotionBlur(p=.2),
            #     albu.MedianBlur(blur_limit=3, p=.3),
            #     albu.Blur(blur_limit=3, p=.5),
            # ], p=.4),
            # albu.OneOf([
            #     albu.RandomContrast(p=.5),
            #     albu.RandomBrightness(p=.5),
            # ], p=.4),
            # albu.ShiftScaleRotate(shift_limit=.0, scale_limit=.45, rotate_limit=45, p=.7),
            # albu.OneOf([
            #     albu.OpticalDistortion(p=0.3),
            #     albu.GridDistortion(p=0.2),
            #     albu.ElasticTransform(p=.2),
            # ], p=.6),
            # albu.HueSaturationValue(p=.5)
            ]
        return albu.Compose(train_transform)
    
    def Normalize(self):
        return albu.Compose([
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0
            )
        ])
    
