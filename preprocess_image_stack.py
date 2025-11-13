from image_preprocessing import SplitZImageStack, PreprocessSplitImages, SplitSingleImages
import time

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Functions to preprocess an nd2 z-stack image
# to use for training the UNET model
##########################################

##################################################
# split single stack into multiple images
z_stck_path = "images"
output_dir = "processed_zstack"
SplitZImageStack(z_stck_path, output_dir)

##################################################
# preprocess the images
start_time = time.time()

output_path = "images/preprocessed"
input_path = "images/processed_zstack"
PreprocessSplitImages(input_path, output_path)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocessing took {elapsed_time:.6f} seconds.")

##################################################
# tile the images further
img_dir = "./images/preprocessed"
output_dir = "./images/subdivided"

SplitSingleImages(img_dir, output_dir)

