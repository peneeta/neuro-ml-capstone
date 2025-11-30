import numpy as np
from scipy.ndimage import gaussian_filter
from cv2_rolling_ball import subtract_background_rolling_ball
import matplotlib.pyplot as plt
from pathlib import Path
import nd2
import tifffile
import os
import cv2
import time

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

def SplitSingleImages(img_dir, output_dir, tile_size=576):
    
    """
    Tile one image into smaller images. Values to try: 384, 576, 768, 1152
    Img files are 2304 x 2304
    """
    
    # select all image files from img_filepath
    os.makedirs(output_dir, exist_ok=True)
    img_dir = Path(img_dir)
    img_files = list(img_dir.glob("*.tif"))
    
    if not img_files:
        print(f"No .tif files found in {img_dir}")
        return
    
    total_tiles = 0
    
    # iterate over image files
    for img_filepath in img_files:
        print(f"PROCESSING {img_filepath}")
        img = tifffile.imread(img_filepath)
        
        if len(img.shape) != 3 and img.shape[0] != 4:
            print("Skipping")
            print("Curr img shape:", img.shape)
            continue
        
        else:
            print("IMAGE SHAPE", img.shape)
            _, height, width = img.shape
            
            # original basename
            base_name = os.path.splitext(os.path.basename(img_filepath))[0]
            
            tiles_x = width // tile_size
            tiles_y = height // tile_size
            
            tile_number = 0
            
            # tile the image
            for y in range(tiles_y):
                for x in range(tiles_x):
                    
                    # tile boundaries
                    top = y * tile_size
                    left = x * tile_size
                    bottom = top + tile_size
                    right = left + tile_size
                    
                    # get tile
                    tile = img[:, top:bottom, left:right]
                    
                    # output name
                    output_filename = f"{base_name}_part_{tile_number}.tif"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # save im
                    tifffile.imwrite(output_path, tile)
                    
                    tile_number += 1
            
            total_tiles += tile_number
        
    print(f"Finished - created {total_tiles} tiles")

def PreprocessSplitImages(img_filepath, output_dir = "preprocessed"):

    # convert to Path object for easier handling
    img_dir = Path(img_filepath)
    
    # get all .tif files in the directory
    tif_files = list(img_dir.glob("*.tif"))
    
    # check if there are any
    if not tif_files:
        print(f"No .tif files found in {img_filepath}")
        return
    
    print(f"Found {len(tif_files)} img file(s)")
    
    # make output dir if not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # iterate and preprocess all files
    for im in tif_files:
        print(f"\nProcessing: {im.name}")
        
        start_time = time.time()
        
        # Open file - tifffile handles multi-channel images
        curr_im = tifffile.imread(im)
        
        # preprocess
        processed = PreprocessImage(curr_im)
        
        print("PROCESSED IMG SHAPE:", processed.shape)
        
        # Save to file with prefix _pr.tif within the output_dir
        output_filename = im.stem + "_pr.tif"
        output_path = output_dir / output_filename
        tifffile.imwrite(output_path, processed, compression="deflate")
        print(f"Saved: {output_filename}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("took ", elapsed_time, " seconds")
    
    print("\nProcessing complete! Saved images")

def GaussianBlur(img):

    blurred_image = np.zeros_like(img)
    
    for c in range(4):
        blurred_image[c] = gaussian_filter(img[c], sigma=1)
    
    return blurred_image

def ZScoreNorm(img):
    """Perform z score normalization across the image
    """
    
    img = img.astype(np.float64)
    corrected = np.empty_like(img)
    
    # iter over channels (CHANNELS ARE THE FIRST DIM)
    for c in range(img.shape[0]):
        
        channel = img[c, ...]
    
        # scale all pixel values to floats between 0 and 1
        mean = np.mean(channel)
        std = np.std(channel)
        
        # in case the channel is empty just return it
        if std == 0:
            corrected[c, ...] = channel
        else:
            normalized = (channel - mean) / std
            corrected[c, ...] = normalized
    
    return corrected
    
def FlatFieldCorrection(img, sigma_xy=200,
                       clip_percent: float = 0.5,
                       correction_strength: float = 0.5,
                       plot=False) -> np.ndarray:
    """
    Applies flat-field (illumination) correction to a multi-channel image.
    Normalizes illumination across the image.
    """
    
    img_float = img.astype(np.float32)
    corrected = np.empty_like(img_float)
    eps = 1e-6

    for c in range(img.shape[0]):
        channel = img_float[c, ...]
        
        channel_range = np.percentile(channel, 99) - np.percentile(channel, 1)
        channel_mean = np.mean(channel)
        
        # skip if channel doesn't have a ton of signal
        if channel_range < channel_mean * 0.1 or channel_range < 1:
            corrected[c, ...] = channel
            if plot:
                print(f"Channel {c}: Skipped (range={channel_range:.2f}, mean={channel_mean:.2f})")
            continue

        # Estimate smooth illumination field - Gaussian w large radius
        pad = int(sigma_xy)
        padded = np.pad(channel, pad_width=pad, mode='reflect')
        field = gaussian_filter(padded.astype(np.float64), sigma=sigma_xy, mode='nearest')
        field = field[pad:-pad, pad:-pad]
        
        # Optional: Clip extreme outliers in the field
        if clip_percent > 0:
            lo = np.percentile(field, clip_percent)
            hi = np.percentile(field, 100 - clip_percent)
            field = np.clip(field, lo, hi)

        # Normalize field
        field_mean = np.mean(field)
        field_norm = field / (field_mean + eps)  # Now centered around 1.0
        
        # limit normalization (prevent large corrections)
        field_norm = np.clip(field_norm, 0.2, 5.0)  # Allow 5x boost, 5x reduction
        
        # Apply correction with strength blending
        # When strength=1.0: divide by field (full correction)
        # When strength=0.0: divide by 1.0 (no correction)
        field_blend = correction_strength * field_norm + (1.0 - correction_strength) * 1.0
        
        corrected_channel = channel / field_blend
        
        # Scale to preserve original intensity range
        orig_mean = np.mean(channel)
        corrected_mean = np.mean(corrected_channel)
        corrected_channel = corrected_channel * (orig_mean / (corrected_mean + eps))
        
        # final clipping
        if np.issubdtype(img.dtype, np.integer):
            corrected_channel = np.clip(corrected_channel, 0, np.iinfo(img.dtype).max)
        else:
            corrected_channel = np.clip(corrected_channel, 0, None)
        
        corrected[c, ...] = corrected_channel
        
        if plot:
            print(f"Channel {c}: Corrected (range={channel_range:.2f})")
            plot_flat_field_correction(channel, field_norm, corrected_channel)

    return corrected
        
def CLAHEContrastAdjustment(img, clip_limit = 2.0, tile_size = 8):
    """
    Enhances contrast of a multi-channel image after flat-field correction.
    
    Parameters
    ----------
    img : np.ndarray
        Input image of shape (C, H, W)
    clip_limit : float
    tile_size : int
    
    Returns
    -------
    np.ndarray
        Contrast-enhanced image
    """
    enhanced = np.empty_like(img)
    
    for c in range(img.shape[0]):
        channel = img[c, ...]
        
        # Normalize to 0-1 for CLAHE
        c_min, c_max = np.percentile(channel, [0.1, 99.9])
        if c_max - c_min < 1e-6:
            enhanced[c, ...] = channel
            continue
            
        channel_norm = np.clip((channel - c_min) / (c_max - c_min), 0, 1)
        
        # Convert to uint16 for CLAHE (better precision than uint8)
        channel_uint = (channel_norm * 65535).astype(np.uint16)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        enhanced_uint = clahe.apply(channel_uint)
        
        # Convert back to original scale
        enhanced_norm = enhanced_uint.astype(np.float32) / 65535.0
        enhanced[c, ...] = enhanced_norm * (c_max - c_min) + c_min
            
    
    # Final clipping to valid range
    if np.issubdtype(img.dtype, np.integer):
        enhanced = np.clip(enhanced, 0, np.iinfo(img.dtype).max)
    else:
        enhanced = np.clip(enhanced, 0, None)
    
    return enhanced.astype(img.dtype)

def ColumnBackgroundSub(image, smooth_sigma=0):
    """
    Remove vertical stripe artifacts by subtracting column-wise background.
    Processes each channel independently.
    
    Parameters:
    -----------
    image : numpy.ndarray
        4D input image with shape (C, H, W) where:
        - C = number of channels
        - H = height
        - W = width
    smooth_sigma : float
        Gaussian smoothing applied to column profiles before subtraction.
        Higher values (e.g., 5-10) create smoother corrections.
        Use 0 for no smoothing.
    
    Returns:
    --------
    numpy.ndarray
        Corrected image with shape (C, H, W) with vertical stripes removed

    """
    
    img_float = image.astype(np.float32)
    n_channels = image.shape[0]
    
    # Process each channel independently
    corrected = np.zeros_like(img_float)
    
    for c in range(n_channels):
        channel = img_float[c, :, :]
        
        # Compute column-wise median background
        col_background = np.median(channel, axis=0)
        
        # Optional: smooth the column background profile
        if smooth_sigma > 0:
            col_background = gaussian_filter(col_background, sigma=smooth_sigma)
        
        # Subtract column background from each column
        corrected[c, :, :] = channel - col_background[np.newaxis, :]
        
        # Shift to non-negative to preserve intensity relationships
        # for downstream processing that expects non-negative values
        min_val = corrected[c, :, :].min()
        if min_val < 0:
            corrected[c, :, :] = corrected[c, :, :] - min_val
    
    return corrected

def BilateralDenoise(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to a multi-channel image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image with shape (C, H, W) where C is number of channels
    d : int, optional (default=5)
        Diameter of each pixel neighborhood used during filtering.
        Large values (>5) are slow. For real-time applications, use d=5.
        For offline processing with better quality, use d=9.
    sigma_color : float, optional (default=75)
        Filter sigma in the color space. Larger value means colors farther apart
        will be mixed together, resulting in larger areas of semi-equal color.
        Typical range: 10-150 for 8-bit images, scale proportionally for float images.
    sigma_space : float, optional (default=75)
        Filter sigma in the coordinate space. Larger value means pixels farther away
        will influence each other as long as their colors are similar.
        Typical range: 10-150
    
    Returns:
    --------
    filtered_image : numpy.ndarray
        Filtered image with same shape as input (C, H, W)
    
    Notes:
    ------
    - Each channel is filtered independently to preserve channel-specific features
    - For fluorescence microscopy, you may want to tune sigma_color based on your
      SNR and typical intensity ranges per channel
    - If your image is normalized to [0,1], scale sigma_color accordingly (e.g., 0.1-0.3)
    """
    # Validate input shape
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (C, H, W), got shape {image.shape}")
    
    n_channels, _, _ = image.shape
    
    # Initialize output array
    process_image = image.astype(np.float32)
    filtered_image = np.zeros_like(process_image)
    
    # Apply bilateral filter to each channel independently
    for c in range(n_channels):
        
        # Extract single channel (H, W)
        channel = process_image[c]
        
        # Apply bilateral filter
        filtered_channel = cv2.bilateralFilter(
            channel, 
            d=d, 
            sigmaColor=sigma_color, 
            sigmaSpace=sigma_space
        )
        
        filtered_image[c] = filtered_channel
    
    return filtered_image

def MultiRollingBallBGSub(image):
    """
    Perform rolling ball background subtraction on each channel.
    
    Parameters
    ----------
    image : ndarray of shape (4, H, W)
        Input 4-channel image
    
    Returns
    -------
    corrected : ndarray of shape (4, H, W)
        Background-subtracted image for each channel
    background : ndarray of shape (4, H, W)
        Estimated background for each channel
    """
    n_channels = image.shape[0]
    corrected = np.zeros_like(image)
    background = np.zeros_like(image)
    
    for i in range(n_channels):
        print("Processing channel", i)
        
        # Convert to uint8
        channel = image[i]
        if channel.dtype in [np.float32, np.float64]:
            if channel.max() <= 1.0:
                channel = (channel * 255).astype(np.uint8)
            else:
                channel = channel.astype(np.uint8)
        elif channel.dtype != np.uint8:
            channel = channel.astype(np.uint8)
        
        # This function returns TWO values: (corrected_image, background)
        corrected[i], background[i] = subtract_background_rolling_ball(
            channel, 
            radius=50, 
            light_background=False
        )
    
    return corrected, background

# FULL PREPROCESSING PIPELINE
def PreprocessImage(full_img, plot=False):
    """
    Function to preprocess a single image

    Args:
        full_img (hxwxn numpy array where n is the number of channels): a single stacked image

    Returns:
        preprocessed img
    """
    
    
    # verify that channels are first dim
    if full_img.shape[0] != 4:
        full_img = np.moveaxis(full_img, -1, 0)
    
    # rm column artifacts, fn converts to np.float32
    # note - shifts to nonnegative values for ffcorr
    corr_img = ColumnBackgroundSub(full_img)
    
    # flat-field correction, fn converts to np.float32
    corr_img = FlatFieldCorrection(corr_img, plot=plot)
    
    # bilateral denoise, fn converts to np.float32
    corr_img = BilateralDenoise(corr_img)
    
    # subtract bg, fn converts to uint8
    corr_img, _ = MultiRollingBallBGSub(corr_img)
   
    # contrast boost, uint16
    corr_img = CLAHEContrastAdjustment(corr_img)
    
    # gaussian blur
    corr_img = GaussianBlur(corr_img)
    
    # norm zscore
    return  ZScoreNorm(corr_img)


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