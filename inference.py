from model import NeuroUNET

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Helpers for model inference
##########################################


def TileImages(img, tile_size=256):
    """Tile one image into smaller images. Values to try: 384, 576, 768, 1152
    Img files are 2304 x 2304

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

def StitchTiles(tile_set):
    pass

def PredictImage(img):
    pass

def PredictImagesFromDir(img_dir):
    pass

# test set dir


