from torch.utils.data import Dataset
import numpy as np
import tifffile
import random

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Custom dataset instantation
##########################################

class EMDataset(Dataset):
    """
    Custom Dataset for loading and tiling images for UNET training.
    """
    def __init__(self, image_paths, tile_size=256, stride=None, augment=False):
        """
        Args:
            image_paths: List of paths to input images
            tile_size: Size of square tiles to extract (default: 256x256)
            stride: Stride for tile extraction. If None, uses tile_size (no overlap)
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        self.augment = augment
        
        # Calculate tile positions for each image
        self.tiles = self._compute_tile_positions()
    
    def _compute_tile_positions(self):
        """Pre-compute all valid tile positions across all images."""
        tiles = []
        for img_idx, img_path in enumerate(self.image_paths):
            # Load image to get dimensions
            img = tifffile.imread(img_path)
            c, h, w = img.shape
            
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

        # load image (4 ch) set ch as first dim
        img = tifffile.imread(img_path)

        # get tile
        x, y = tile_info['x'], tile_info['y']
        tile = img[:, y:y + self.tile_size, x:x + self.tile_size]

        # Apply augmentation if enabled
        if self.augment:
            tile = self._augment(tile)
        
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
        img = np.rot90(img, k, axes=(1, 2))  # Specify axes for h,w dimensions
        
        return img.copy()