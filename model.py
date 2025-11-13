import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import tifffile
import wandb

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# NeuroUNET model and training/validation
# functions
##########################################

class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class NeuroUNET(nn.Module):
    """UNET with 3 encoder layers, 1 bottleneck, and 3 decoder layers"""
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 512 because of concatenation
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 because of concatenation
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # 128 because of concatenation
        
        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        return x

def dice_loss(pred, target, smooth=1e-6):
    """
    Calculate Dice loss for segmentation tasks
    
    Args:
        pred: Predicted output from model
        target: Ground truth target
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value (1 - Dice coefficient)
    """
    # Flatten the tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    
    # Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1 - dice_coeff

def TrainModel(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device='cuda'):
    """
    Train the UNET model
    
    Args:
        model: UNET model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
    """
    
    model = model.to(device)
    # criterion = nn.L1Loss()  # MAE loss - REMOVED
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    # add wandb for plotting
    wandb.init(
        project="NeuroUNET Testing",
        config={
            "model_name": "NeuroUNET",
            "learning_rate": 6e-4,
            "scheduler_step_size": 3,
            "scheduler_gamma": 0.1,
            "epochs": 10,
            "loss_function": "dice_loss",
        }
    )
    
    for epoch in range(num_epochs):
        ##### TRAINING #####
        model.train()
        train_loss = 0.0
        
        prog = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, images in enumerate(prog):
            images = images.to(device)
            
            # Split the 4-channel image: first 2 channels as input, last 2 as target
            inputs = images[:, :2, :, :]  # First 2 channels
            targets = images[:, 2:, :, :]  # Last 2 channels
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss using Dice loss
            loss = dice_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch
            })
            
            # Update progress bar with current loss
            prog.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        ##### VALIDATION #####
        model.eval()
        val_loss = 0.0
        
        # tqdm progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images in val_pbar:
                images = images.to(device)
                inputs = images[:, :2, :, :]
                targets = images[:, 2:, :, :]
                
                outputs = model(inputs)
                loss = dice_loss(outputs, targets)
                val_loss += loss.item()
                
                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # TODO: fix LR I think
        wandb.log({
                "val/loss": avg_val_loss,
                "learning_rate": lr,
                "epoch": epoch
        })
        
        # print summary
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}')
        
        # save best model state dict
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f'saved best state dict for epoch {epoch+1}')
    
    wandb.finish()
    
    print('Training completed!')
    return model