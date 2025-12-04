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
# PreprocessSplitImages(input_path, output_path)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Preprocessing took {elapsed_time:.6f} seconds.")

# base_fp = Path("/run/user/1000/gvfs/smb-share:server=zhao-nas.lan.local.cmu.edu,share=zhao-lab/Magnify Biosciences/capstone/11Nov25_acquire_40x_z_2x2")

# output_base = Path("/run/user/1000/gvfs/smb-share:server=zhao-nas.lan.local.cmu.edu,share=zhao-lab/Magnify Biosciences/capstone/11Nov25_acquire_40x_z_2x2/")


##################################################
# tile the images further
# base_fp = Path("/run/user/1000/gvfs/smb-share:server=zhao-nas.lan.local.cmu.edu,share=zhao-lab/Magnify Biosciences/capstone/11Nov25_acquire_40x_z_2x2")

base_fp = home / "em_capstone_f25" 

img_dir  = base_fp / "Images" / "A1_preprocessed"
output_dir = base_fp / "Images" / "A1_tiled"

print("Splitting Images")
start_time = time.time()

# just try locally for now, TODO replace with stride 100 
SplitInformativePatches(
        img_dir,
        output_dir,
        tile_size=576,
        stride = 200,
        tissue_threshold=0.01,
        nucleus_threshold=0.9, # looks like this is the best to include nuclei
        receptor_threshold=0.01,
        min_variance=100.0
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Took {elapsed_time:.6f} seconds.")

