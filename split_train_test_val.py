import os
import shutil
import random
from pathlib import Path

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Split images into separate dirs for train/test/val
##########################################

def SplitTrainTestVal(image_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Split images in a directory into train/val/test sets.
    
    Args:
        image_dir: Path to directory containing images
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.2)
        seed: Random seed for reproducibility (default: 42)
    """
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
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
    
    # Calculate split indices
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split into sets
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    
    # Create directories
    train_dir = image_dir / 'train'
    val_dir = image_dir / 'val'
    test_dir = image_dir / 'test'
    
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(exist_ok=True)
    
    # Move images
    for img in train_imgs:
        shutil.move(str(img), str(train_dir / img.name))
    
    for img in val_imgs:
        shutil.move(str(img), str(val_dir / img.name))
    
    for img in test_imgs:
        shutil.move(str(img), str(test_dir / img.name))
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"  Test:  {len(test_imgs)} images")

#############################################################################

# base_fp = Path("/run/user/1000/gvfs/smb-share:server=zhao-nas.lan.local.cmu.edu,share=zhao-lab/Magnify Biosciences/capstone/11Nov25_acquire_40x_z_2x2")

home = Path.home()
base_fp = home / "em_capstone_f25" 
dir_to_split = base_fp / "Images" / "A1_tiled"

# Test with A1 first
SplitTrainTestVal(dir_to_split)