import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import wandb
import pywt
from pytorch_msssim import ssim

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# NeuroUNET model and training/validation
# functions
##########################################

class DWT(nn.Module):
    """Discrete Wavelet Transform - replaces MaxPool for better detail preservation"""
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        batch, channels, height, width = x.shape
        
        # Apply DWT to each channel separately
        coeffs_list = []
        for b in range(batch):
            batch_coeffs = []
            for c in range(channels):
                # Get single channel
                img = x[b, c].cpu().detach().numpy()
                
                # Apply 2D DWT
                coeffs = pywt.dwt2(img, self.wavelet)
                cA, (cH, cV, cD) = coeffs
                
                # Stack all coefficients: LL, LH, HL, HH
                # This preserves all information
                stacked = torch.stack([
                    torch.from_numpy(cA).to(x.device),
                    torch.from_numpy(cH).to(x.device),
                    torch.from_numpy(cV).to(x.device),
                    torch.from_numpy(cD).to(x.device)
                ], dim=0)
                batch_coeffs.append(stacked)
            
            coeffs_list.append(torch.stack(batch_coeffs, dim=0))
        
        # Reshape: (B, C*4, H/2, W/2)
        result = torch.stack(coeffs_list, dim=0)
        result = result.view(batch, channels * 4, height // 2, width // 2)
        
        return result

class IWT(nn.Module):
    """Inverse Wavelet Transform - replaces ConvTranspose for better detail reconstruction"""
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
    
    def forward(self, x):
        # x shape: (B, C*4, H, W) where C*4 contains LL, LH, HL, HH coefficients
        batch, channels_x4, height, width = x.shape
        channels = channels_x4 // 4
        
        # Reconstruct each channel
        recon_list = []
        for b in range(batch):
            batch_recon = []
            for c in range(channels):
                # Extract the 4 wavelet coefficients for this channel
                cA = x[b, c * 4 + 0].cpu().detach().numpy()
                cH = x[b, c * 4 + 1].cpu().detach().numpy()
                cV = x[b, c * 4 + 2].cpu().detach().numpy()
                cD = x[b, c * 4 + 3].cpu().detach().numpy()
                
                # Apply inverse DWT
                coeffs = (cA, (cH, cV, cD))
                img = pywt.idwt2(coeffs, self.wavelet)
                
                batch_recon.append(torch.from_numpy(img).to(x.device))
            
            recon_list.append(torch.stack(batch_recon, dim=0))
        
        # Reshape: (B, C, H*2, W*2)
        result = torch.stack(recon_list, dim=0)
        
        return result

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
    def __init__(self, in_channels=2, out_channels=2, wavelet='haar'):
        super().__init__()
        
        self.dwt = DWT(wavelet=wavelet)
        
        # Encoder (3 levels instead of 4)
        self.enc1 = DoubleConv(in_channels, 64)
        self.adapt1 = nn.Conv2d(64 * 4, 64, kernel_size=1)
        
        self.enc2 = DoubleConv(64, 128)
        self.adapt2 = nn.Conv2d(128 * 4, 128, kernel_size=1)
        
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder with DWT
        enc1 = self.enc1(x)  # 128×128
        x = self.adapt1(self.dwt(enc1))  # 64×64
        
        enc2 = self.enc2(x)  # 64×64
        x = self.adapt2(self.dwt(enc2))  # 32×32
        
        x = self.bottleneck(x)  # 32×32
        
        # Decoder
        x = self.up2(x)  # 64×64
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)  # 128×128
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return self.out(x)  # 128×128

    def predict(self, image, patch_size=128):
        """
        Predict on an image by dividing it into non-overlapping patches. Output image of shape (C, H, W) with predicted channels 1,2
        """
        c, h, w = image.shape
        
        # Calculate padding needed
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        # Pad the image 
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        else:
            padded = image
        
        _, padded_h, padded_w = padded.shape
        
        # Calculate number of patches
        n_patches_h = padded_h // patch_size
        n_patches_w = padded_w // patch_size
        
        # Initialize output (keep all 4 channels)
        output_padded = np.zeros((4, padded_h, padded_w), dtype=image.dtype)
        
        # Copy input channels 0 and 3 to output
        output_padded[0, :, :] = padded[0, :, :]
        output_padded[3, :, :] = padded[3, :, :]
        
        # Process each patch
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                
                # Extract patch
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                
                patch = padded[:, y_start:y_end, x_start:x_end]
                
                # Extract channels 0 and 3 for model input
                patch_input = patch[[0, 3], :, :]
                
                # Predict channels 1 and 2
                patch_pred = self.predict(patch_input)
                
                # Place predicted channels into output
                output_padded[1:3, y_start:y_end, x_start:x_end] = patch_pred
        
        # Remove padding to get back to original size
        output = output_padded[:, :h, :w]
        
        return output
      
def TrainModel(model, train_loader, val_loader, checkpoint_dir, dapi_channel = 0, cb_channel = 3, num_epochs=20, lr=1e-3, device='cuda'):
    """
    Train the UNET model with learning rate scheduling
    
    model: UNET model instance
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    num_epochs: Number of training epochs
    lr: Initial learning rate
    device: Device to train on ('cuda' or 'cpu')
    """
    
    # init the checkpoint dir
    checkpoint_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(torch.device(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    # add wandb for plotting
    wandb.init(
        project="NeuroUNET with Real Data",
        config={
            "model_name": "NeuroUNET",
            "learning_rate": lr,
            "epochs": num_epochs,
            "loss_function": "SSIM",
        }
    )
        
    for epoch in range(num_epochs):
            ##### TRAINING #####
            model.train()
            train_loss = 0.0
            
            prog = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            
            for batch_idx, images in enumerate(prog):
                images = images.to(device)
                
                input_channels = [dapi_channel, cb_channel]
                all_channels = [0, 1, 2, 3]
                
                target_channels = [c for c in all_channels if c not in input_channels]

                # divide into input and target
                inputs = images[:, input_channels, :, :]
                targets = images[:, target_channels, :, :]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # SSIM loss
                ssim_value = ssim(outputs, targets, data_range=1.0, size_average=True)
                loss = 1 - ssim_value
                
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
                    
                    input_channels = [dapi_channel, cb_channel]
                    all_channels = [0, 1, 2, 3]
                    
                    target_channels = [c for c in all_channels if c not in input_channels]

                    # divide into input and target
                    inputs = images[:, input_channels, :, :]
                    targets = images[:, target_channels, :, :]
                    
                    outputs = model(inputs)
                    
                    # loss SSIM
                    ssim_value = ssim(outputs, targets, data_range=1.0, size_average=True)
                    loss = 1 - ssim_value
                    val_loss += loss.item()
                    
                    # update progress bar with current loss
                    val_pbar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
            # update scheduler
            scheduler.step(avg_val_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            wandb.log({
                    "val/loss": avg_val_loss,
                    "train/epoch_loss": avg_train_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch
            })
            
            # print summary
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, '
                f'LR: {current_lr:.2e}')
            
            # save best model state dict
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), checkpoint_dir / 'best_unet_model.pth')
                print(f'saved best state dict for epoch {epoch+1}')
        
    wandb.finish()
        
    print('Training completed!')
    return model