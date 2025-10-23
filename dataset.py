import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random


# Dataloader for training the model

class EMDataset(Dataset):
    """
    Custom Dataset for loading and tiling images for UNET training.
    """
    def __init__(self, image_paths, tile_size=256, stride=None, 
                 preprocess_fn=None, augment=False):
        """
        Args:
            image_paths: List of paths to input images
            tile_size: Size of square tiles to extract (default: 256x256)
            stride: Stride for tile extraction. If None, uses tile_size (no overlap)
            preprocess_fn: Custom preprocessing function to apply to images
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.preprocess_fn = preprocess_fn
        self.augment = augment
        
        # Calculate tile positions for each image
        self.tiles = self._compute_tile_positions()
    
    def _compute_tile_positions(self):
        """Pre-compute all valid tile positions across all images."""
        tiles = []
        for img_idx, img_path in enumerate(self.image_paths):
            # Load image to get dimensions
            img = Image.open(img_path)
            w, h = img.size
            
            # Calculate number of tiles
            for y in range(0, h - self.tile_size + 1, self.stride):
                for x in range(0, w - self.tile_size + 1, self.stride):
                    tiles.append({
                        'img_idx': img_idx,
                        'x': x,
                        'y': y
                    })
        return tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        """Load and return a single tile."""
        tile_info = self.tiles[idx]
        img_path = self.image_paths[tile_info['img_idx']]
        
        # Load image (4 ch)
        img = Image.open(img_path).convert('RGBA')
        
        # Extract tile
        x, y = tile_info['x'], tile_info['y']
        tile = img.crop((x, y, x + self.tile_size, y + self.tile_size))
        
        # Convert to numpy array
        tile = np.array(tile)
        
        # Apply custom preprocessing
        if self.preprocess_fn is not None:
            tile = self.preprocess_fn(tile)
        
        # Apply augmentation if enabled
        if self.augment:
            tile = self._augment(tile)
        
        # Convert to torch tensor and normalize to [0, 1]
        if tile.dtype == np.uint8:
            tile = tile.astype(np.float32) / 255.0
        
        # Change from HWC to CHW format
        tile = torch.from_numpy(tile).permute(2, 0, 1).float()
        
        return tile
    
    def _augment(self, img):
        """Simple augmentation: random flips and rotations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
        
        # Random 90-degree rotation
        k = random.randint(0, 3)
        img = np.rot90(img, k)
        
        return img.copy()


