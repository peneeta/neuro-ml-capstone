import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import tifffile
from pathlib import Path
import wandb
import pywt

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

class GlobalContextBranch(nn.Module):
    """
    Processes the full 576x576 image to extract global context features.
    Uses progressive downsampling to create a compact feature vector.
    """
    def __init__(self, in_channels=2, feature_dim=256):
        super().__init__()
        
        # Progressive downsampling: 576 -> 288 -> 144 -> 72 -> 36 -> 18 -> 9
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 288x288
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 144x144
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 72x72
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 36x36
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling to get feature vector
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Project to desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True)
        )

class NeuroUNET(nn.Module):
    """UNET with DWT/IWT for superior detail preservation
    
    Key improvements:
    - DWT replaces MaxPool: preserves all frequency information
    - IWT replaces ConvTranspose: perfect reconstruction
    - Channel adaptation layers handle 4x channel expansion from DWT
    """
    def __init__(self, in_channels=2, out_channels=2, wavelet='haar', context_dim=256):
        super().__init__()
        
        # DWT/IWT transforms
        self.dwt = DWT(wavelet=wavelet)
        self.iwt = IWT(wavelet=wavelet)
        
        # Global context
        self.global_branch = GlobalContextBranch(in_channels, context_dim)
        
        # Context injection layers - add global features to decoder
        self.context_proj_bottleneck = nn.Linear(context_dim, 512)
        self.context_proj_dec3 = nn.Linear(context_dim, 256)
        self.context_proj_dec2 = nn.Linear(context_dim, 128)
        
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
    
    def forward(self, x_patch, x_full):
        # Full image features with global conteext
        global_context = self.global_branch(x_full)
        
        # Encoder with DWT downsampling
        enc1 = self.enc1(x_patch)
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
        
        # Project context and add to bottleneck features
        context_bottleneck = self.context_proj_bottleneck(global_context)
        # Reshape and add: (B, 512) -> (B, 512, 1, 1) and broadcast
        context_bottleneck = context_bottleneck.view(x.size(0), -1, 1, 1)
        x = x + context_bottleneck
        
        # Decoder with IWT upsampling
        x = self.expand3(x)
        x = self.iwt(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
    
        # inject global context
        context_dec3 = self.context_proj_dec3(global_context)
        context_dec3 = context_dec3.view(x.size(0), -1, 1, 1)
        x = x + context_dec3
        
        x = self.expand2(x)
        x = self.iwt(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        # inject global context
        context_dec2 = self.context_proj_dec2(global_context)
        context_dec2 = context_dec2.view(x.size(0), -1, 1, 1)
        x = x + context_dec2
        
        x = self.expand1(x)
        x = self.iwt(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        return x

def TrainModel(model, train_loader, val_loader, checkpoint_dir, dapi_channel = 0, cb_channel = 3, num_epochs=20, lr=1e-3, device='cuda'):
    """
    Train the UNET model with learning rate scheduling
    
    Args:
        model: UNET model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Initial learning rate
        device: Device to train on ('cuda' or 'cpu')
        scheduler_type: Type of scheduler ('plateau', 'step', 'cosine')
    """
    
    # init the checkpoint dir
    checkpoint_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    # add wandb for plotting
    wandb.init(
        project="NeuroUNET Testing",
        config={
            "model_name": "NeuroUNET",
            "learning_rate": lr,
            "epochs": num_epochs,
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
                
                input_channels = [dapi_channel, cb_channel]
                all_channels = [0, 1, 2, 3]
                
                target_channels = [c for c in all_channels if c not in input_channels]

                # divide into input and target
                inputs = images[:, input_channels, :, :]
                targets = images[:, target_channels, :, :]
                
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
                    
                    input_channels = [dapi_channel, cb_channel]
                    all_channels = [0, 1, 2, 3]
                    
                    target_channels = [c for c in all_channels if c not in input_channels]

                    # divide into input and target
                    inputs = images[:, input_channels, :, :]
                    targets = images[:, target_channels, :, :]
                    
                    outputs = model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets)
                    val_loss += loss.item()
                    
                    # Update progress bar with current loss
                    val_pbar.set_postfix({'loss': loss.item()})
            
            avg_val_loss = val_loss / len(val_loader)
            
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