from torch.utils.data import Dataset
import numpy as np
import tifffile
import random
import torch

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Custom dataset instantation
##########################################

class EMDataset(Dataset):
    """
    Custom Dataset for loading and tiling images for UNET training
    Each epoch uses one random tile per image
    """
    def __init__(self, image_paths, tile_size=256, augment=False):
        """
        Args:
            image_paths: List of paths to input images
            tile_size: Size of square tiles to extract (default: 256x256)
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.tile_size = tile_size
        self.augment = augment
        
        # Store image dimensions for computing valid tile positions
        self.image_dims = self._load_image_dimensions()
        
        # Random tile positions (one per image) - will be regenerated each epoch
        self.current_tiles = self._generate_random_tiles()
    
    def _load_image_dimensions(self):
        """Load and cache image dimensions."""
        dims = []
        for img_path in self.image_paths:
            img = tifffile.imread(img_path)
            c, h, w = img.shape
            dims.append({'c': c, 'h': h, 'w': w})
        return dims
    
    def _generate_random_tiles(self):
        """Generate one random tile position for each image."""
        tiles = []
        for img_idx, dims in enumerate(self.image_dims):
            h, w = dims['h'], dims['w']
            
            # Calculate valid ranges for tile positions
            max_y = h - self.tile_size
            max_x = w - self.tile_size
            
            # Ensure we have valid positions
            if max_y < 0 or max_x < 0:
                raise ValueError(f"Image {img_idx} is smaller than tile_size {self.tile_size}")
            
            # Random position for this image
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            tiles.append({
                'img_idx': img_idx,
                'x': x,
                'y': y
            })
        return tiles
    
    def regenerate_tiles(self):
        """Call this at the start of each epoch to get new random tiles."""
        self.current_tiles = self._generate_random_tiles()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and return a single tile."""
        tile_info = self.current_tiles[idx]
        img_path = self.image_paths[tile_info['img_idx']]

        # load image (4 ch) set ch as first dim
        img = tifffile.imread(img_path)

        # get tile
        x, y = tile_info['x'], tile_info['y']
        tile = img[:, y:y + self.tile_size, x:x + self.tile_size]

        # Apply augmentation if enabled
        if self.augment:
            tile = self._augment(tile)
            
        # use torch tensor instead of numpy arr    
        tile = torch.from_numpy(tile).float()
        
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