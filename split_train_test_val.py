import os
import shutil
import random
from pathlib import Path
import tifffile as tiff

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Split images into separate dirs for train/test/val
##########################################

def SplitTwoSets(image_dir, train_ratio=0.8, test_ratio=0.2, seed=42):
    """
    Split images in a directory into train_full/test sets.
    
    Args:
        image_dir: Path to directory containing images
        train_ratio: Proportion for training set (default: 0.8)
        test_ratio: Proportion for test set (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    """

    # Convert to Path object
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Common image extensions
    img_extensions = {'.tif'}
    
    # Get all image files
    images = [f for f in image_dir.iterdir() 
              if f.is_file() and f.suffix.lower() in img_extensions]
    
    if not images:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Shuffle images
    random.seed(seed)
    random.shuffle(images)
    
    # Calculate split index
    n = len(images)
    train_end = int(n * train_ratio)
    
    # Split into sets
    train_imgs = images[:train_end]
    test_imgs = images[train_end:]
    
    # Create directories
    train_dir = image_dir / 'train_full'
    test_dir = image_dir / 'test'
    
    for d in [train_dir, test_dir]:
        d.mkdir(exist_ok=True)
    
    # Move images
    for img in train_imgs:
        shutil.move(str(img), str(train_dir / img.name))
    
    for img in test_imgs:
        shutil.move(str(img), str(test_dir / img.name))
    
    print(f"Dataset split complete:")
    print(f"  Train_full: {len(train_imgs)} images")
    print(f"  Test:       {len(test_imgs)} images")
    
def TileImages(input_dir, output_dir, tile_size=256, stride=200):
    """
    Split 4-channel TIF images into tiles and save to output directory.
    
    Args:
        input_dir: Path to directory containing input images
        output_dir: Path to directory where tiles will be saved
        tile_size: Size of square tiles (default: 256)
        stride: Stride between tiles in pixels (default: 200)
    """
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all TIF files
    tif_files = list(input_dir.glob('*.tif'))
    
    if not tif_files:
        raise ValueError(f"No TIF files found in {input_dir}")
    
    total_tiles = 0
    
    for img_path in tif_files:
        # Read image using tifffile
        img_array = tiff.imread(img_path)
        
        # Get dimensions (assuming shape is [channels, height, width])
        channels, height, width = img_array.shape
        
        # Get base filename without extension
        base_name = img_path.stem
        
        tile_count = 0
        
        # Iterate over image with stride
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                # Extract tile (all channels, tile_size x tile_size)
                tile = img_array[:, y:y+tile_size, x:x+tile_size]
                
                # Create filename with position information
                tile_name = f"{base_name}_tile_{tile_count}_y{y}_x{x}.tif"
                tile_path = output_dir / tile_name
                
                # Save tile using tifffile
                tiff.imwrite(tile_path, tile)
                
                tile_count += 1
        
        total_tiles += tile_count
        print(f"Processed {img_path.name}: {tile_count} tiles created")
    
    print(f"\nTotal tiles created: {total_tiles}")
    print(f"Tiles saved to: {output_dir}")

#############################################################################

# we first split the full images
home = Path.home()
base_fp = home / "lm_lab_proj" 

dirs = ["B1", "B2", "B3"]

for dir_name in dirs:
    dir_to_split = base_fp / "preprocessed" / (dir_name + "_preprocessed")
    SplitTwoSets(dir_to_split)
    
# then subdivide into tiles
for dir_name in dirs:
    dir_to_split = base_fp / "preprocessed" / (dir_name + "_preprocessed/train_full")
    output_dir = base_fp / "for_training" / (dir_name + "_split")
    
    # tile images (stride 200 px)
    TileImages(dir_to_split, output_dir)
    
    # validation set will be 10%
    SplitTwoSets(output_dir, train_ratio = 0.9)
     