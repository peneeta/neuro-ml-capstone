import os
from PIL import Image
import numpy as np
import tifffile as tiff

from scipy.ndimage import gaussian_filter

# PROPOSED FLOW
# Gaussian background subtraction -> Hist matching -> Tile Images -> Local Denoising -> normalize px between 0 and 1 values -> RETURN IMAGE STACK

def NormalizeImageChannels(img):
    """Scales pixel values in a 2D img between 0 and 1

    Args:
        img (2d numpy array): numpy array representing a single channel

    Returns:
        img_norm: the scaled 2D image with intensity values between 0 and 1
    """
    
    # scale all pixel values to floats between 0 and 1
    img_min = img.min()
    img_max = img.max()
    
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    
    return img_norm


def SelectActiveChannel(img):
    """Select the color channel that is nonzero assuming unstacked (i.e. if using individual stain images like DAPI only or CB only)

    Args:
        img (np.ndarray): 

    Returns:
        img: nonzero 2D image channel
    """
    
    # if only 2 dims, return as-is
    if img.ndim == 2:
        return img, 0
    
    # move channels to first dim for iteration: (C, H, W)
    if img.shape[0] not in (1, 2, 3, 4):
        img = np.moveaxis(img, -1, 0)

    # return the first non-empty channel
    for i, ch in enumerate(img):
        if not np.all(ch == 0):
            
            # return with dim 1 as last dim: (H, W, 1)
            ch = np.expand_dims(ch, axis=-1)
            return ch, i


# looks like sigma = 0.5 is okay
def BackgroundSubtraction(img, sigma = 0.5):
    # TODO this function is acting strange, could probably use code from ZL python file here
    # ensure type
    # img = img.astype(np.float32)

    # Gaussian smoothing to estimate background
    background = gaussian_filter(img, sigma=sigma)

    # Subtract background
    subtracted = background - img

    # Clip negative values (optional, depends on use case)
    subtracted = np.clip(subtracted, 0, None)

    return subtracted, background

    

# TODO USE THIS FUNC FOR A SINGLE CHANNEL 
def NormalizeSingleChannel(img):
    active_ch = SelectActiveChannel(img)
    
    
    pass


# img files are 2304 x 2304
def TileImages(img, tile_size=768):
    """Tile one image into smaller images. Values to try: 384, 576, 768, 1152

    Args:
        img (numpy array): 2304x2304 tissue image
        tile_size (int, optional): Size of image. Defaults to 768.
    """
    
    h, w, n = img.shape
    tiled_imgs = []
    
    # if tile size not divisible, print this
    if h % tile_size != 0 or w % tile_size != 0:
        print(f"Warning: Image size ({h}, {w}) not divisible by {tile_size}. "
              f"Last tiles will include remaining pixels.")

    # iterate with step = tile_size, but ensure last tile covers all remaining pixels
    y_starts = list(range(0, h, tile_size))
    x_starts = list(range(0, w, tile_size))

    for y in y_starts:
        y_end = min(y + tile_size, h)
        
        for x in x_starts:
            x_end = min(x + tile_size, w)
            tile = img[y:y_end, x:x_end, :]
            tiled_imgs.append(tile)

    return tiled_imgs

