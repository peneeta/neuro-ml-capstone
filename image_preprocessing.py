import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import nd2
import tifffile

##########################################
# Image preprocessing code
##########################################

# TODO: TEST SPLIT Z-STACK
# TODO: TEST FULL PIPELINE

def SplitZImageStack(img_filepath):
    """Splits images in a Z-stack into single images, saves them as TIF files
    
    Args:
        img_filepath: Path to directory containing .nd2 files
    """
    # convert to Path object for easier handling
    img_dir = Path(img_filepath)
    
    # get all .nd2 files in the directory
    nd2_files = list(img_dir.glob("*.nd2"))
    
    if not nd2_files:
        print(f"No .nd2 files found in {img_filepath}")
        return
    
    print(f"Found {len(nd2_files)} .nd2 file(s)")
    
    # make output dir
    output_dir = img_dir / "processed_zstack"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each ND2 file
    for nd2_file in nd2_files:
        print(f"\nProcessing: {nd2_file.name}")
        
        # Load the ND2 file
        with nd2.ND2File(nd2_file) as f:
            # Get the image stack
            stack = f.asarray()
            
            # Get base filename without extension
            base_name = nd2_file.stem
            
            # Determine the number of slices
            # Stack shape can vary, but Z is typically the first dimension
            if stack.ndim == 2:
                # Single slice image
                n_slices = 1
                stack = stack[None, ...]  # Add dimension for consistent processing
            elif stack.ndim == 3:
                # Z-stack
                n_slices = stack.shape[0]
            else:
                # Handle multi-dimensional data (e.g., Z, C, Y, X)
                print(f"  Warning: Image has shape {stack.shape}. Assuming first dimension is Z.")
                n_slices = stack.shape[0]
            
            print(f"  Found {n_slices} slice(s)")
            
            # Save each slice as a separate TIF file
            for i in range(n_slices):
                # Format slice number with leading zeros (01, 02, etc.)
                slice_num = str(i + 1).zfill(2)
                
                # Create output filename
                output_name = f"{base_name}_{slice_num}.tif"
                output_path = output_dir / output_name
                
                # Extract the slice
                if stack.ndim == 3:
                    slice_img = stack[i]
                else:
                    # For higher dimensional data, take the first slice along first dimension
                    slice_img = stack[i, ...]
                
                # Save as TIF
                tifffile.imwrite(output_path, slice_img)
                
                if (i + 1) % 10 == 0 or i == n_slices - 1:
                    print(f"  Saved {i + 1}/{n_slices} slices")
    
    print("\nProcessing complete! Saved image")

def NormalizeImageChannels(img):
    """Scales pixel values in a 2D img between 0 and 1
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
    for c in range(img.shape[2]):
        channel = img[..., c]
        
        # find the background for this channel
        bg = np.percentile(channel, low_perc)

        # for debugging
        bg_images.append(bg)

        # subtract the background from this channel
        corrected = channel - bg
        corrected[corrected < 0] = 0.0
        out[..., c] = corrected
    
        # Plot if requested
    if plot:
        plot_background_subtraction(img, bg_images, out)

    # return bg-subbed image
    return out

def FlatFieldCorrection(img, sigma_xy = 200,
                       clip_percent: float = 1.0, plot = False) -> np.ndarray:
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

        # estimate smooth illumination field and pad to reduce edge fall-off
        pad = sigma_xy
        padded = np.pad(channel, pad_width=pad, mode='reflect')
        field = gaussian_filter(padded, sigma=sigma_xy, mode='nearest')
        field = field[pad:-pad, pad:-pad]

        # Clamp extremes (fix the bug)
        lo = np.percentile(field, clip_percent)
        hi = np.percentile(field, 100 - clip_percent)
        field = np.clip(field, lo, hi)

        # Normalize field so center region has value ~1.0
        # This means dividing by field keeps center unchanged, boosts edges
        H, W = channel.shape
        central_field = field[H//4:3*H//4, W//4:3*W//4]
        field_mean = np.mean(central_field)
        
        # Now: center ≈ 1.0, dark edges < 1.0
        field_norm = field / (field_mean + eps)
        
        # Divide: center stays same, edges get brightened
        corrected_channel = channel / np.maximum(field_norm, eps)
        corrected_channel = np.clip(corrected_channel, 0, 255)
        
        # fix scaling
        corrected[..., c] = np.clip(corrected_channel, 0, np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else None)
        
        # DEBUGGING PRINT - Visualize the field, pre and post correction
        if plot:
            plot_flat_field_correction(channel, field_norm, corrected_channel)

    return corrected


# img files are 2304 x 2304
def TileImages(img, tile_size=768):
    """Tile one image into smaller images. Values to try: 384, 576, 768, 1152
    
    NOTE - THIS FUNCTION IS NOT BEING USED (SEE DATALOADER)

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


def PreprocessImage(full_img):
    
    # select active channel per single stained img
    active_ch = SelectActiveChannel(full_img)
    
    # background sub with stack (h, w, n)
    bg_subbed = BackgroundSubtraction(active_ch)
    
    # flat-field correction
    ff_corr = FlatFieldCorrection(bg_subbed)
    
    # normalize between 0 and 1
    norm_img = NormalizeImageChannels(ff_corr)
    
    return norm_img





### DEBUG PRINT FUNCS ###

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

    n_channels = original.shape[2]
    
    # Create figure with 4 rows (Original, Background, Corrected) 
    # and n_channels columns
    fig, axes = plt.subplots(3, n_channels, figsize=(6*n_channels, 16))
    
    # Handle single channel case
    if n_channels == 1:
        axes = axes.reshape(-1, 1)
    
    print("\n=== Background Subtraction Analysis ===")
    
    for c in range(n_channels):
        # Original image
        im0 = axes[0, c].imshow(original[..., c], cmap='gray')
        axes[0, c].set_title(f'Original - Channel {c}', fontsize=12)
        axes[0, c].axis('off')
        plt.colorbar(im0, ax=axes[0, c], fraction=0.046, pad=0.04)
        
        # Background visualization (uniform image at bg value)
        bg_img = np.full_like(original[..., c], bg_values[c], dtype=float)
        im1 = axes[1, c].imshow(bg_img, cmap='gray', vmin=original[..., c].min(), vmax=original[..., c].max())
        axes[1, c].set_title(f'Background Value: {bg_values[c]:.2f}', fontsize=12)
        axes[1, c].axis('off')
        plt.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.04)
        
        # Corrected image
        im2 = axes[2, c].imshow(corrected[..., c], cmap='gray')
        axes[2, c].set_title(f'Corrected - Channel {c}', fontsize=12)
        axes[2, c].axis('off')
        plt.colorbar(im2, ax=axes[2, c], fraction=0.046, pad=0.04)
        
        # Print statistics
        orig_min = original[..., c].min()
        orig_max = original[..., c].max()
        orig_mean = original[..., c].mean()
        corr_min = corrected[..., c].min()
        corr_max = corrected[..., c].max()
        corr_mean = corrected[..., c].mean()
        
        print(f"\nChannel {c}:")
        print(f"  Background value (percentile): {bg_values[c]:.2f}")
        print(f"  Original - Min: {orig_min:.2f}, Max: {orig_max:.2f}, Mean: {orig_mean:.2f}")
        print(f"  Corrected - Min: {corr_min:.2f}, Max: {corr_max:.2f}, Mean: {corr_mean:.2f}")
        print(f"  Mean shift: {orig_mean - corr_mean:.2f} ({100*(orig_mean - corr_mean)/orig_mean:.2f}%)")
        print(f"  Background as % of max: {100*bg_values[c]/orig_max:.2f}%")
    
    plt.tight_layout()
    plt.show()