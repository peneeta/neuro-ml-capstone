from model import NeuroUNET
import numpy as np
import os
from scipy.stats import pearsonr
from pathlib import Path
import csv
import tifffile as tiff

##########################################
# NeuroML Capstone Project
# CMU Fall 2025
# Helpers for model inference
##########################################

def PearsonCorrelationTwoImages(ground_truth, predicted):
    """
    Compute Pearson correlation between ground truth and predicted images for channels 1 and 2.
    
    Parameters:
    -----------
    ground_truth : np.ndarray
        Ground truth image of shape (4, H, W)
    predicted : np.ndarray
        Predicted image of shape (4, H, W)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'channel_1': Pearson r for channel 1
        - 'channel_2': Pearson r for channel 2
        - 'overall': Average Pearson r across both channels
    """
    # Extract channels 1 and 2 and flatten
    pred_ch1 = predicted[1, :, :].flatten()
    pred_ch2 = predicted[2, :, :].flatten()
    
    gt_ch1 = ground_truth[1, :, :].flatten()
    gt_ch2 = ground_truth[2, :, :].flatten()
    
    # Compute Pearson correlation for each channel
    r_ch1, _ = pearsonr(gt_ch1, pred_ch1)
    r_ch2, _ = pearsonr(gt_ch2, pred_ch2)
    
    # Compute overall correlation
    overall_r = (r_ch1 + r_ch2) / 2
    
    return {
        'channel_1': r_ch1,
        'channel_2': r_ch2,
        'overall': overall_r
    }


def PredictTestSet(model, input_dir, output_dir, patch_size=128):
    """
    Run model prediction on all images in a directory and save results.
    
    Parameters:
    -----------
    model : object
        Model object with a predict method
    input_dir : str
        Directory containing input images
    output_dir : str
        Directory to save predicted images
    patch_size : int
        Patch size for prediction (default: 128)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_path.glob(f'*.tif'))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        print(f"Processing {img_file.name}...")
        
        # Load image
        image = tiff.imread(img_file)
        
        # Predict
        predicted = model.predict(image, patch_size=patch_size)
        
        # Create output filename with _pred postfix
        base_name = img_file.stem  # filename without extension
        output_filename = f"{base_name}_pred.tif"
        output_filepath = output_path / output_filename
        
        # Save predicted image
        tiff.imwrite(output_filepath, predicted)
        print(f"Saved to {output_filepath}")
    
    print(f"Completed processing {len(image_files)} images")


def PearsonCorrelationTestSet(ground_truth_dir, predicted_dir, output_csv, pred_postfix='_pred'):
    """
    Calculate Pearson correlations between ground truth and predicted images.
    
    Parameters:
    -----------
    ground_truth_dir : str
        Directory containing ground truth images
    predicted_dir : str
        Directory containing predicted images
    output_csv : str
        Path to output CSV file
    pred_postfix : str
        Postfix added to predicted filenames (default: '_pred')
    """
    gt_path = Path(ground_truth_dir)
    pred_path = Path(predicted_dir)
    
    # Get all ground truth image files
    gt_files = list(gt_path.glob(f'*.tif'))
    
    results = []
    
    print(f"Found {len(gt_files)} ground truth images")
    
    for gt_file in gt_files:
        base_name = gt_file.stem
        
        # Construct predicted filename
        pred_filename = f"{base_name}{pred_postfix}.tif"
        pred_file = pred_path / pred_filename
        
        if not pred_file.exists():
            print(f"Warning: Predicted file not found for {gt_file.name}, skipping...")
            continue
        
        print(f"Calculating correlation for {base_name}...")
        
        # Load images
        ground_truth = tiff.imread(gt_file)
        predicted = tiff.imread(pred_file)
        
        # Compute correlation
        correlation = PearsonCorrelationTwoImages(ground_truth, predicted)
        
        # Store results
        results.append({
            'filename': base_name,
            'channel_1_correlation': correlation['channel_1'],
            'channel_2_correlation': correlation['channel_2'],
            'overall_correlation': correlation['overall']
        })
    
    # Write results to CSV
    if results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'channel_1_correlation', 'channel_2_correlation', 'overall_correlation']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        # Calculate and print summary statistics
        ch1_mean = np.mean([r['channel_1_correlation'] for r in results])
        ch2_mean = np.mean([r['channel_2_correlation'] for r in results])
        overall_mean = np.mean([r['overall_correlation'] for r in results])
        
        print(f"\nResults saved to {output_csv}")
        print(f"Processed {len(results)} image pairs")
        print(f"\nSummary Statistics:")
        print(f"Channel 1 Mean Correlation: {ch1_mean:.4f}")
        print(f"Channel 2 Mean Correlation: {ch2_mean:.4f}")
        print(f"Overall Mean Correlation: {overall_mean:.4f}")
    else:
        print("No matching image pairs found!")

