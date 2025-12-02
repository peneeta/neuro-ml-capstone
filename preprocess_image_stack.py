from image_preprocessing import PreprocessSplitImages, SplitInformativePatches
import time
from pathlib import Path

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Functions to preprocess an nd2 z-stack image
# to use for training the UNET model
##########################################

home = Path.home()

##################################################
# preprocess the images
start_time = time.time()

# A1
# output_path = home / "em_capstone_f25"/ "Images"/ "A1_preprocessed"
# input_path = home / "em_capstone_f25" / "Images" / "A1"
PreprocessSplitImages(input_path, output_path)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocessing took {elapsed_time:.6f} seconds.")


##################################################
# # tile the images further
img_dir = "./images/preprocessed"
output_dir = "./images/subdivided"

img_dir  = home / "em_capstone_f25"/ "Images"/ "A1_preprocessed"
output_dir = home / "em_capstone_f25" / "Images" / "A1_tiled"

print("Splitting Images")
start_time = time.time()

SplitInformativePatches(
        img_dir,
        output_dir,
        tile_size=576,
        tissue_threshold=0.5,
        nucleus_threshold=0.9, # looks like this is the best to include nuclei
        receptor_threshold=0.01,
        min_variance=100.0
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Took {elapsed_time:.6f} seconds.")

