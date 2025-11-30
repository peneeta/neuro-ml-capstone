from image_preprocessing import SplitZImageStack, PreprocessSplitImages, SplitSingleImages
import time

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Functions to preprocess an nd2 z-stack image
# to use for training the UNET model
##########################################


##################################################
# preprocess the images
start_time = time.time()

# A1
output_path = "~/em_capstone_f25/images/A1_preprocessed"
input_path = "~/em_capstone_f25/images/A1"
PreprocessSplitImages(input_path, output_path)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Preprocessing took {elapsed_time:.6f} seconds.")


##################################################
# tile the images further
img_dir = "./images/preprocessed"
output_dir = "./images/subdivided"

SplitSingleImages(img_dir, output_dir)

