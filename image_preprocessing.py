import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import nd2
import tifffile

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Image Preprocessing Code
##########################################

def SplitZImageStack(img_filepath, output_dir = "processed_zstack"):
    """
    Splits images in a Z-stack into single images, saves them as TIF files
    Z stack image has dims (Z, num_channels, h, w)
    For this project: Image has shape (118, 4, 2304, 2304)
    
    FIX FOR MEMORY:
    Uses lazy loading to read one slice at a time to avoid memory issues.
    
    Args:
        img_filepath: Path to directory containing .nd2 files
    """
    # convert to Path object for easier handling
    img_dir = Path(img_filepath)
    
    # get all .nd2 files in the directory
    nd2_files = list(img_dir.glob("*.nd2"))
    
    # check if there are any
    if not nd2_files:
        print(f"No .nd2 files found in {img_filepath}")
        return
    
    print(f"Found {len(nd2_files)} .nd2 file(s)")
    
    # make output dir if not exist
    output_dir = img_dir / output_dir
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each ND2 file
    for nd2_file in nd2_files:
        print(f"\nProcessing: {nd2_file.name}")
        
        # Open the ND2 file (but don't load into memory)
        with nd2.ND2File(nd2_file) as f:
 
            base_name = nd2_file.stem
            shape = f.shape
            print(f"  Image shape: {shape}")
            
            # figure out numslices
            if len(shape) == 2:
                n_slices = 1
            else:
                n_slices = shape[0]
            
            print(f"  Found {n_slices} slice(s)")
            
            # Save each slice as a separate TIF file
            for i in range(n_slices):
                # slice number with leading zeros (01, 02, etc.)
                slice_num = str(i + 1).zfill(2)
                
                # output filename
                output_name = f"{base_name}_{slice_num}.tif"
                output_path = output_dir / output_name
                
                # check numslices
                if len(shape) == 2:
                    slice_img = f.asarray()
                else:
                    # LAZY LOADING: use dask array indexing
                    slice_img = f.to_dask()[i].compute()

                # TIF with compression
                tifffile.imwrite(output_path, slice_img, compression='zlib')
                
                # delete the slice to free memory immediately
                del slice_img
                
                if (i + 1) % 10 == 0 or i == n_slices - 1:
                    print(f"  Saved {i + 1}/{n_slices} slices")
    
    print("\nProcessing complete! Saved images")

def NormalizeImageChannels(img):
    """Scales pixel values between 0 and 1
    """
    
    img = img.astype(np.float32)
    corrected = np.empty_like(img)
    
    # iter over channels
    for c in range(img.shape[0]):
        
        channel = img[c, ...]
    
        # scale all pixel values to floats between 0 and 1
        c_min = channel.min()
        c_max = channel.max()
        
        c_norm = (channel - c_min) / (c_max - c_min + 1e-8)
        
        corrected[c, ...] = c_norm
    
    return corrected

def SelectActiveChannel(img):
    """Select the color channel that is nonzero assuming unstacked (i.e. if using individual stain images like DAPI only or CB only)
    Not included in the pipeline since stacked img has 4 channels combined already

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
    
    # move channels to first dim (C, H, W)
    if img.shape[0] not in (1, 2, 3, 4):
        img = np.moveaxis(img, -1, 0)
    
    
    print(img.shape)

    # return the first non-empty channel
    for i, ch in enumerate(img):
        if not np.all(ch == 0):
            return ch.reshape(1, img.shape[1], img.shape[2]), i

def ZScoreNorm(img):
    """Perform z score normalization across the image
    """
    
    img = img.astype(np.float32)
    corrected = np.empty_like(img)
    
    # iter over channels (CHANNELS ARE THE FIRST DIM)
    for c in range(img.shape[0]):
        
        channel = img[c, ...]
    
        # scale all pixel values to floats between 0 and 1
        mean = np.mean(channel)
        std = np.std(channel)
        
        if std == 0:
            return np.zeros_like(channel)
        
        normalized = (channel - mean) / std
        corrected[c, ...] = normalized
    
    return corrected
    
def BackgroundSubtraction(img, low_perc = 1.0, plot=False):
    """ (Method from Zhao Lab)

    Args:
        img (numpy arr): Subtracts background per channel with percentile method
        low_perc (float, optional): Percentile (0–100) used to estimate background per channel. Typical values: 0.5–2.0 for dense tissue. Defaults to 0.5.

    Returns:
        numpy arr: background-subtracted image (bg is removed PER channel)
    """
    
    # empty out arr
    out = np.empty_like(img)
    
    bg_images = []

    # with channels as last dim
    for c in range(img.shape[0]):
        channel = img[c, ...]
        
        # find the background for this channel
        bg = np.percentile(channel, low_perc)

        # for debugging
        bg_images.append(bg)

        # subtract the background from this channel
        corrected = channel - bg
        corrected[corrected < 0] = 0.0
        out[c, ...] = corrected
    
        # Plot if requested
    if plot:
        plot_background_subtraction(img, bg_images, out)

    # return bg-subbed image
    return out

def OldFlatFieldCorrection(img, sigma_xy = 200,
                       clip_percent: float = 1.0, plot = False) -> np.ndarray:
    """
    Applies flat-field (illumination) correction to a multi-channel image.
    OLD VERSION - odd results with the image stack, field is not smooth

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
    eps = 1e-7 # avoid dividing by 0

    for c in range(img.shape[0]):
        channel = img[c, ...]
        
        # SKIP EMPTY CHANNELS OR LOW-CONTRAST
        # Check if channel has meaningful signal (not just noise)
        channel_range = np.percentile(channel, 99) - np.percentile(channel, 1)
        channel_mean = np.mean(channel)
        
        # If the signal is very weak relative to mean (likely empty/noise), skip correction
        if channel_range < channel_mean or channel_range < 1:
            corrected[c, ...] = channel
            if plot:
                print(f"Channel {c}: Skipped (range={channel_range:.2f}, mean={channel_mean:.2f})")
            continue

        # estimate smooth illumination field and pad to reduce edge fall-off
        pad = sigma_xy
        padded = np.pad(channel, pad_width=pad, mode='reflect')
        field = gaussian_filter(padded, sigma=sigma_xy, mode='reflect')
        field = field[pad:-pad, pad:-pad]

        # Clamp extremes (fix the bug)
        # lo = np.percentile(field, clip_percent)
        # hi = np.percentile(field, 100 - clip_percent)
        # field = np.clip(field, lo, hi)
        
        median = np.median(field)
        mad = np.median(np.abs(field - median))
        field = np.clip(field, median - 3*mad, median + 3*mad)

        # Normalize field so center region has value ~1.0
        # This means dividing by field keeps center unchanged, boosts edges
        H, W = channel.shape
        # central_field = field[H//4:3*H//4, W//4:3*W//4]
        field_mean = np.mean(field)
        
        # Now: center ≈ 1.0, dark edges < 1.0
        field_norm = field / (field_mean + eps)
        
        # Divide: center stays same, edges get brightened
        corrected_channel = channel / np.maximum(field_norm, eps)
        corrected_channel = np.clip(corrected_channel, 0, 255)
        
        # fix scaling
        corrected[c, ...] = np.clip(corrected_channel, 0, np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else None)
        
        # DEBUGGING PRINT - Visualize the field, pre and post correction
        if plot:
            plot_flat_field_correction(channel, field_norm, corrected_channel)

    return corrected

def FlatFieldCorrection(img, sigma_xy=200,
                       clip_percent: float = 1.0, 
                       correction_strength: float = 0.5,
                       plot=False) -> np.ndarray:
    """
    Applies flat-field (illumination) correction to a multi-channel image.
    Brightens dark regions while preserving bright regions.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), where C is the number of channels.
    sigma_xy : float, optional
        Gaussian smoothing sigma (in pixels) to estimate illumination field.
    clip_percent : float, optional
        Percentile used to clamp the illumination field to remove extremes.
    correction_strength : float, optional
        Strength of correction (0-1). 1.0 = full correction, 0.5 = half correction.
    plot : bool, optional
        Whether to plot the correction process.

    Returns
    -------
    np.ndarray
        Flat-field corrected image of same shape (H, W, C).
    """
    
    img_float = img.astype(np.float32)
    corrected = np.empty_like(img_float)
    eps = 1e-6

    for c in range(img.shape[0]):
        channel = img_float[c, ...]
        
        # Check if channel has meaningful signal (not just noise)
        channel_range = np.percentile(channel, 99) - np.percentile(channel, 1)
        channel_mean = np.mean(channel)
        
        # If the signal is very weak relative to mean (likely empty/noise), skip correction
        if channel_range < channel_mean * 0.1 or channel_range < 1:
            corrected[c, ...] = channel
            if plot:
                print(f"Channel {c}: Skipped (range={channel_range:.2f}, mean={channel_mean:.2f})")
            continue

        # Estimate smooth illumination field with proper padding
        pad = sigma_xy
        padded = np.pad(channel, pad_width=pad, mode='reflect')
        field = gaussian_filter(padded.astype(np.float64), sigma=sigma_xy, mode='nearest')
        field = field[pad:-pad, pad:-pad]
        
        # Clamp extremes (fix the bug)
        lo = np.percentile(field, clip_percent)
        hi = np.percentile(field, 100 - clip_percent)
        field = np.clip(field, lo, hi)

        # Normalize field to [0, 1] range where 1.0 = brightest areas
        field_max = np.percentile(field, 99.9)
        field_min = np.percentile(field, 0.1)
        
        if field_max - field_min < eps:
            corrected[c, ...] = channel
            continue
            
        field_norm = (field - field_min) / (field_max - field_min + eps)
        # field_norm now ranges from ~0 (darkest) to ~1 (brightest)
        
        # Prevent division by very small numbers (limit boost to 5x max)
        field_norm = np.clip(field_norm, 0.2, 1.0)
        
        # Apply partial correction using correction_strength
        # correction_strength = 1.0: full correction (divide by field)
        # correction_strength = 0.0: no correction (divide by 1.0)
        field_corrected = 1.0 / field_norm
        field_blend = correction_strength * (field_corrected - 1.0) + 1.0
        
        corrected_channel = channel * field_blend
        
        # Clip to reasonable range - don't exceed 1.2x the original 99th percentile
        orig_p99 = np.percentile(channel, 99)
        corrected_channel = np.clip(corrected_channel, 0, orig_p99 * 1.2)
        
        corrected[c, ...] = corrected_channel
        
        # fix scaling
        corrected[c, ...] = np.clip(corrected_channel, 0, np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else None)
        
        if plot:
            print(f"Channel {c}: Corrected (range={channel_range:.2f}, max_boost={np.max(field_blend):.2f}x)")
            plot_flat_field_correction(channel, field_norm, corrected_channel)

    
    return corrected
        
def TileImages(img, tile_size=768):
    """Tile one image into smaller images. Values to try: 384, 576, 768, 1152
    Img files are 2304 x 2304
    
    NOTE - THIS FUNCTION IS NOT BEING USED (SEE DATALOADER)

    Args:
        img (numpy array): 2304x2304 tissue image
        tile_size (int, optional): Size of image. Defaults to 768.
    """
    
    n, h, w = img.shape
    tiled_imgs = []
    
    # if tile size not divisible, print this
    if h % tile_size != 0 or w % tile_size != 0:
        print(f"Warning: Image size ({h}, {w}) not divisible by {tile_size}. "
              f"Last tiles will include remaining pixels.")

    # iterate with step = tile_size
    y_starts = list(range(0, h, tile_size))
    x_starts = list(range(0, w, tile_size))

    for y in y_starts:
        y_end = min(y + tile_size, h)
        
        for x in x_starts:
            x_end = min(x + tile_size, w)
            tile = img[:, x:x_end, y:y_end]
            tiled_imgs.append(tile)

    return tiled_imgs

# FULL PREPROCESSING PIPELINE
def PreprocessImage(full_img, plot=False):
    """
    Function to preprocess a single image

    Args:
        full_img (hxwxn numpy array where n is the number of channels): a single stacked image

    Returns:
        preprocessed img
    """
    
    # background sub with stack (h, w, n)
    bg_subbed = BackgroundSubtraction(full_img, plot=plot)
    
    # flat-field correction
    ff_corr = FlatFieldCorrection(bg_subbed, plot=plot)
    
    # norm zscore and scale
    zsc_norm = ZScoreNorm(ff_corr)
    norm_img = NormalizeImageChannels(zsc_norm)
    
    return norm_img

##########################################
# Debug Prints
##########################################

def plot_flat_field_correction(channel, field_norm, corrected_channel):
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(channel, cmap='gray')
    plt.title('Original Channel')
    
    plt.subplot(132)
    plt.imshow(field_norm, cmap='gray')
    plt.title('Estimated Field (normalized)')
    
    plt.subplot(133)
    plt.imshow(corrected_channel, cmap='gray')
    
    plt.title('Corrected')
    plt.colorbar()
    plt.show()

def plot_background_subtraction(original, bg_values, corrected):

    n_channels = original.shape[0]
    
    # Create figure with 4 rows (Original, Background, Corrected) 
    # and n_channels columns
    fig, axes = plt.subplots(3, n_channels, figsize=(6*n_channels, 16))
    
    # Handle single channel case
    if n_channels == 1:
        axes = axes.reshape(-1, 1)
    
    print("\n=== Background Subtraction Analysis ===")
    
    for c in range(n_channels):
        
        # Original image
        im0 = axes[0, c].imshow(original[c, ...], cmap='gray')
        
        axes[0, c].set_title(f'Original - Channel {c}', fontsize=12)
        axes[0, c].axis('off')
        plt.colorbar(im0, ax=axes[0, c], fraction=0.046, pad=0.04)
        
        # Background visualization (uniform image at bg value)
        bg_img = np.full_like(original[c, ...], bg_values[c], dtype=float)
        im1 = axes[1, c].imshow(bg_img, cmap='gray', vmin=original[c, ...].min(), vmax=original[c, ...].max())
        axes[1, c].set_title(f'Background Value: {bg_values[c]:.2f}', fontsize=12)
        axes[1, c].axis('off')
        plt.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.04)
        
        # Corrected image
        im2 = axes[2, c].imshow(corrected[c, ...], cmap='gray')
        axes[2, c].set_title(f'Corrected - Channel {c}', fontsize=12)
        axes[2, c].axis('off')
        plt.colorbar(im2, ax=axes[2, c], fraction=0.046, pad=0.04)
        
        # Print statistics
        orig_min = original[c, ...].min()
        orig_max = original[c, ...].max()
        orig_mean = original[c, ...].mean()
        corr_min = corrected[c, ...].min()
        corr_max = corrected[c, ...].max()
        corr_mean = corrected[c, ...].mean()
        
        print(f"\nChannel {c}:")
        print(f"  Background value (percentile): {bg_values[c]:.2f}")
        print(f"  Original - Min: {orig_min:.2f}, Max: {orig_max:.2f}, Mean: {orig_mean:.2f}")
        print(f"  Corrected - Min: {corr_min:.2f}, Max: {corr_max:.2f}, Mean: {corr_mean:.2f}")
        print(f"  Mean shift: {orig_mean - corr_mean:.2f} ({100*(orig_mean - corr_mean)/orig_mean:.2f}%)")
        print(f"  Background as % of max: {100*bg_values[c]/orig_max:.2f}%")
    
    plt.tight_layout()
    plt.show()