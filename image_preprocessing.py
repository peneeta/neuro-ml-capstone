import os
from PIL import Image
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter

# TODO: DENOISE IMAGE FUNCTION
# TODO: COMBINE PIPELINE INTO A FUNCTION

def NormalizeImageChannels(img_list):
    """Scales pixel values in a 2D img between 0 and 1
    """
    norm_img_list = []
    
    for img in img_list:
        
        # scale all pixel values to floats between 0 and 1
        img_min = img.min()
        img_max = img.max()
        
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)
        
        norm_img_list.append(img_norm)
    
    return norm_img_list


def SelectActiveChannel(img):
    """Select the color channel that is nonzero assuming unstacked (i.e. if using individual stain images like DAPI only or CB only)

    Args:
        img (np.ndarray): 

    Returns:
        img: nonzero 2D image channel
    """
    # set type here
    img = img.astype(np.float32, copy=False)
    
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


def BackgroundSubtraction(img, low_perc = 1.0):
    """ (Method from Zhao Lab)

    Args:
        img (numpy arr): Subtracts background per channel with percentile method
        low_perc (float, optional): Percentile (0–100) used to estimate background per channel. Typical values: 0.5–2.0 for dense tissue. Defaults to 0.5.

    Returns:
        numpy arr: background-subtracted image (bg is removed PER channel)
    """
    
    # empty out arr
    out = np.empty_like(img)

    # with channels as last dim
    for c in range(img.shape[2]):
        channel = img[..., c]
        
        # find the background for this channel
        try:
            bg = np.percentile(channel, low_perc)
        except Exception:
            bg = 0.0

        # subtract the background from this channel
        corrected = channel - bg
        corrected[corrected < 0] = 0.0
        out[..., c] = corrected

    # return bg-subbed image
    return out

def FlatFieldCorrection(img, sigma_xy = 16,
                       clip_percent: float = 1.0) -> np.ndarray:
    """
    Applies flat-field (illumination) correction to a multi-channel image.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), where C is the number of channels.
    sigma_xy : float, optional
        Gaussian smoothing sigma (in pixels) to estimate illumination field.
    clip_percent : float, optional
        Percentile used to clamp the illumination field to remove extremes.

    Returns
    -------
    np.ndarray
        Flat-field corrected image of same shape (H, W, C), dtype float32.
    """
    
    corrected = np.empty_like(img)
    eps = 1e-6 # avoid dividing by 0

    for c in range(img.shape[2]):
        channel = img[..., c]

        # estimate smooth illumination field
        # pad to reduce edge fall-off
        pad = sigma_xy
        padded = np.pad(channel, pad_width=pad, mode='reflect')
        field = gaussian_filter(padded, sigma=sigma_xy, mode='nearest')
        field = field[pad:-pad, pad:-pad]

        # clamp extremes to mitigate outlier influence
        lo = np.percentile(field, clip_percent)
        hi = np.percentile(field, clip_percent)
        field = np.clip(field, lo, hi)
        
        # normalize using mean of central region to avoid dim edges
        H, W = channel.shape
        central_field = field[H//4:3*H//4, W//4:3*W//4]
        m = np.percentile(central_field, 90)
        if m <= 0:
            m = eps
        field = field / m

        # apply correction and clip negatives
        corrected[..., c] = np.clip(channel / np.maximum(field, eps), 0, None)

    return corrected


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

def DenoiseImage(img):
    pass


# TODO 
def NormalizeImage(img):
    
    # select active channel per single stained img
    # stack images in order
    # background sub with stack (h, w, n)
    
    # tile images with stack (h, w, n)
    # denoise each tile
    # normalize between 0 and 1
    # return tiles
    pass



