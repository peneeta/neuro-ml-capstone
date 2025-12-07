from image_preprocessing import PreprocessSplitImages
import time
from pathlib import Path

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Functions to preprocess an images for training
##########################################

base_fp = Path("/ocean/projects/cis250266p/pwojcik")

# preprocess the images
start_time = time.time()

dirs_to_process = ["B1", "B2", "B3"]

for dir_name in dirs_to_process:
    start_time = time.time()
    
    input_path = base_fp / "original_images" / dir_name
    output_path = base_fp / "preprocessed" / f"{dir_name}_preprocessed"
    
    PreprocessSplitImages(input_path, output_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Preprocessing {dir_name} took {elapsed_time:.6f} seconds")
