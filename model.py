import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import tifffile
import wandb
import pywt

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# NeuroUNET model and training/validation
# functions
##########################################

# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
# incorporate these loss funcs

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
    """UNET with DWT/IWT for superior detail preservation
    
    Key improvements:
    - DWT replaces MaxPool: preserves all frequency information
    - IWT replaces ConvTranspose: perfect reconstruction
    - Channel adaptation layers handle 4x channel expansion from DWT
    """
    def __init__(self, in_channels=2, out_channels=2, wavelet='haar'):
        super().__init__()
        
        # DWT/IWT transforms
        self.dwt = DWT(wavelet=wavelet)
        self.iwt = IWT(wavelet=wavelet)
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.adapt1 = nn.Conv2d(64 * 4, 64, kernel_size=1)
        
        self.enc2 = DoubleConv(64, 128)
        self.adapt2 = nn.Conv2d(128 * 4, 128, kernel_size=1)
        
        self.enc3 = DoubleConv(128, 256)
        self.adapt3 = nn.Conv2d(256 * 4, 256, kernel_size=1)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (upsampling path)
        # Need to expand channels before IWT (C -> C*4)
        self.expand3 = nn.Conv2d(512, 256 * 4, kernel_size=1)
        self.dec3 = DoubleConv(512, 256)  # 512 because of concatenation
        
        self.expand2 = nn.Conv2d(256, 128 * 4, kernel_size=1)
        self.dec2 = DoubleConv(256, 128)
        
        self.expand1 = nn.Conv2d(128, 64 * 4, kernel_size=1)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder with DWT downsampling
        enc1 = self.enc1(x)
        x = self.dwt(enc1)
        x = self.adapt1(x)
        
        enc2 = self.enc2(x)
        x = self.dwt(enc2)
        x = self.adapt2(x)
        
        enc3 = self.enc3(x)
        x = self.dwt(enc3)
        x = self.adapt3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with IWT upsampling
        x = self.expand3(x)
        x = self.iwt(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.expand2(x)
        x = self.iwt(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.expand1(x)
        x = self.iwt(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        return x

def TrainModel(model, train_loader, val_loader, num_epochs=20, lr=1e-3, device='cuda'):
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    # add wandb for plotting
    wandb.init(
        project="NeuroUNET Testing",
        config={
            "model_name": "NeuroUNET",
            "learning_rate": 1e-3,
            "scheduler_step_size": 3,
            "scheduler_gamma": 0.1,
            "epochs": 10,
            "loss_function": "MSE",
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
                
                # Compute loss using MSE loss
                loss = nn.functional.mse_loss(outputs, targets)
                
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
                    loss = nn.functional.mse_loss(outputs, targets)
                    val_loss += loss.item()
                    
                    # Update progress bar with current loss
                    val_pbar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
            wandb.log({
                    "val/loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
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