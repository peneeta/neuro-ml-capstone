import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

def MakeCompositeImage(image, colors, channel_order=None):
    """
    Stack multiple image channels with specified colors.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image array with shape (n_channels, height, width)
    colors : list of str or list of tuples
        Color for each channel. Can be color names ('red', 'cyan') or RGB tuples
    channel_order : list of int, optional
        Order in which to stack channels. If None, uses [1, 2]
        Example: [0, 3, 2, 1] stacks channel 0 first, then 3, then 2, then 1
    
    Returns:
    --------
    numpy.ndarray
        RGB composite image with shape (height, width, 3)
    """
    n_channels = image.shape[0]
    height, width = image.shape[1], image.shape[2]
    
    # Set default channel order to only channels 1 and 2
    if channel_order is None:
        channel_order = [1, 2]
    
    # Validate channel indices
    for idx in channel_order:
        if idx >= n_channels or idx < 0:
            raise ValueError(f"Channel index {idx} is out of bounds for image with {n_channels} channels")
    
    # Initialize composite RGB image
    composite = np.zeros((height, width, 3))
    
    # Stack channels in specified order
    for idx in channel_order:
        # Get channel and normalize to 0-1 if needed
        channel = image[idx]
        if channel.max() > 1:
            channel = channel / channel.max()
        
        # Convert color to RGB tuple
        color_rgb = to_rgb(colors[idx]) if isinstance(colors[idx], str) else colors[idx]
        
        # Apply color to channel and add to composite
        for c in range(3):
            composite[:, :, c] += channel * color_rgb[c]
    
    # Clip values to valid range
    composite = np.clip(composite, 0, 1)
    
    return composite

# Example usage:
# Stack only channels 1 and 2 (default behavior)
# composite = stack_channels(img, ['cyan', 'red', 'limegreen', 'darkblue'])
# plt.imshow(composite)
# plt.show()

# Or specify different channels:
# composite = stack_channels(img, ['cyan', 'red', 'limegreen', 'darkblue'], channel_order=[0, 3])