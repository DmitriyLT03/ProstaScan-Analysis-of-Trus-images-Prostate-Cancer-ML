import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchmetrics import JaccardIndex
from torch import optim
from tqdm import tqdm
from unet import UNet

from utils.CustomDataset import load_data
from utils.loss import JaccardLoss

@torch.inference_mode()
def evaluate(model, dataloader, IoU, device):
    model.eval()
    score = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc= f'Validation', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        mask_pred = model(image)
        col_minus = 0
        for i in mask_pred:
            if i < 0:
                col_minus+=1
        print(col_minus)
        iou_curr = IoU(mask_true, mask_pred)
        score += iou_curr
            
    return iou_curr / max(len(dataloader), 1)

def train_model(
    model,
    optimizer, 
    criterion,
    jaccard, 
    device,
    train_loader,
    val_loader,
    epochs: int=10,
    learning_rate: float=1e-5,
    img_scale: float=0.5, 
):
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    # criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as epoch_bar:
            for batch in train_loader:
                images, mask = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.long)
                
                output = model(images)
                loss = criterion(output, mask.float())
                    
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                train_loss += loss.item()
                epoch_bar.update()
                epoch_bar.set_postfix(**{'loss (batch)': loss.item()})
                
        model.eval()
        val_score = evaluate(model, val_loader, jaccard, device)
        print(f'Valid {epoch} epoch with iou-score:{val_score}')
        scheduler.step(val_score)


if __name__=="__main__":
    
    train_loader, valid_loader = load_data(
        test_size=0.3, batch_size=1, img_size=256, dir='./data/all_data/', artificial_increase=20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    jaccard = JaccardIndex(task='binary', num_classes=1)

    model = UNet(n_channels=1, n_classes=1, bilinear=False)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    # criterion = nn.BCEWithLogitsLoss()

    criterion = JaccardLoss().to(device)
    IoU_val = JaccardLoss(log_loss=True).to(device)
    
    train_model(model, optimizer, criterion, jaccard, device, train_loader, valid_loader, epochs=30)
