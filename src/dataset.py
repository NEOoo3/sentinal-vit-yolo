"""
Indian Road Environment Dataset Pipeline.
Implements specialized augmentations for:
- Delhi-style Haze/Smog
- Monsoon Rain & Road Glare
- High-Density Motion Blur (Unstructured Traffic)
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from .config import Config

class IndianRoadsDataset(Dataset):
    """
    Specialized dataset class for Indian Road Conditions.
    Compatible with UVH-26 and standard YOLO-format annotations.
    """
    
    def __init__(self, config: Config, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.img_size = config.model.input_size
        
        # Define the 'Indian Vibe' Augmentation Pipeline
        self.transform = self._get_transforms()
        
        # Mock data listing - In production, you would crawl your data_root here
        self.image_files = [] 
        
    def _get_transforms(self):
        """
        Builds an augmentation pipeline based on Config.py parameters.
        """
        aug_cfg = self.config.dataset.augmentation
        
        transforms = []
        
        if self.is_train:
            # 1. Weather-based Augmentations (The Indian Specialty)
            transforms.extend([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=aug_cfg.dust_haze, p=aug_cfg.dust_haze),
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=aug_cfg.rain),
                A.MotionBlur(blur_limit=7, p=aug_cfg.motion_blur),
            ])
            
            # 2. Lighting & Color (Noon Glare to Evening Low-light)
            transforms.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            ])
            
            # 3. Geometric (Indian roads have non-standard camera angles)
            transforms.extend([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=aug_cfg.hflip),
            ])

        # 4. Standard Preprocessing
        transforms.extend([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        # Returning a dummy size for initial testing
        return 100

    def __getitem__(self, idx):
        """
        Returns a processed image and its transformed bounding boxes.
        """
        # Create a dummy image for Phase 1/2 integration testing
        image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Mock bounding box for a pedestrian [x_center, y_center, width, height]
        bboxes = [[0.5, 0.5, 0.2, 0.4]]
        class_labels = [0] # 0 = Pedestrian

        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        return transformed['image'], torch.tensor(transformed['bboxes'])

def get_dataloader(config: Config, is_train: bool = True):
    """Utility to create the data loader for training/testing."""
    dataset = IndianRoadsDataset(config, is_train)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=is_train,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory
    )
