#!/usr/bin/env python3
"""
NIfTI Volume Calculator

This script loads a segmented NIfTI file and calculates the volume of each class
based on voxel counts and voxel spacing information.

Requirements:
    - nibabel
    - numpy
    - pandas (optional, for CSV export)

Usage:
    Set the configuration variables below and run the script
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

# Path to your segmented NIfTI file
INPUT_FILE = "/data/aniket/BrainTumorSegmentation/test_dir/BraTS2021_00234/BraTS2021_00234_seg.nii"

# Volume unit: 'mm', 'cm', 'm', 'ml', 'l'
VOLUME_UNIT = "mm"

# Include background class (0) in calculations
INCLUDE_BACKGROUND = False

# Save results to CSV file (set to None to disable)
SAVE_CSV = None  # e.g., "volume_results.csv"

# ============================================================================

def load_nifti(file_path):
    """
    Load NIfTI file and return image data and header
    
    Args:
        file_path (str): Path to NIfTI file
        
    Returns:
        tuple: (image_data, header, affine_matrix)
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        return data, header, affine
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        sys.exit(1)

def get_voxel_volume(header, unit_factor=1.0):
    """
    Calculate voxel volume from NIfTI header
    
    Args:
        header: NIfTI header object
        unit_factor (float): Conversion factor (e.g., 1.0 for mm³, 0.001 for cm³)
        
    Returns:
        float: Volume of single voxel in specified units
    """
    # Get voxel dimensions from header
    pixdim = header.get_zooms()
    
    # Calculate voxel volume (multiply x, y, z dimensions)
    voxel_volume = np.prod(pixdim[:3])  # Only use first 3 dimensions
    
    # Apply unit conversion factor
    voxel_volume *= unit_factor
    
    return voxel_volume

def calculate_class_volumes(segmentation_data, voxel_volume, exclude_background=True):
    """
    Calculate volume for each class in segmentation
    
    Args:
        segmentation_data (numpy.ndarray): 3D segmentation array
        voxel_volume (float): Volume of single voxel
        exclude_background (bool): Whether to exclude class 0 from results
        
    Returns:
        dict: Dictionary with class labels as keys and volumes as values
    """
    # Get unique classes in segmentation
    unique_classes = np.unique(segmentation_data)
    
    # Calculate volumes
    class_volumes = {}
    
    for class_label in unique_classes:
        # Skip background if requested
        if exclude_background and class_label == 0:
            continue
            
        # Count voxels for this class
        voxel_count = np.sum(segmentation_data == class_label)
        
        # Calculate volume
        volume = voxel_count * voxel_volume
        
        class_volumes[int(class_label)] = {
            'voxel_count': int(voxel_count),
            'volume': volume
        }
    
    return class_volumes

def format_volume_results(class_volumes, unit_name="mm³"):
    """
    Format and display volume results
    
    Args:
        class_volumes (dict): Dictionary with class volumes
        unit_name (str): Name of volume unit for display
    """
    print(f"\n{'='*60}")
    print("VOLUME CALCULATION RESULTS")
    print(f"{'='*60}")
    
    total_volume = 0
    
    print(f"{'Class':<8} {'Voxel Count':<12} {'Volume':<15} {'Unit':<8}")
    print("-" * 60)
    
    for class_label, data in sorted(class_volumes.items()):
        voxel_count = data['voxel_count']
        volume = data['volume']
        total_volume += volume
        
        print(f"{class_label:<8} {voxel_count:<12} {volume:<15.3f} {unit_name:<8}")
    
    print("-" * 60)
    print(f"{'Total':<8} {'':<12} {total_volume:<15.3f} {unit_name:<8}")
    print(f"{'='*60}")

def get_unit_factor(unit):
    """
    Get conversion factor for different units
    
    Args:
        unit (str): Target unit ('mm', 'cm', 'm')
        
    Returns:
        tuple: (factor, unit_name)
    """
    unit_factors = {
        'mm': (1.0, 'mm³'),
        'cm': (0.001, 'cm³'),
        'm': (1e-9, 'm³'),
        'ml': (0.001, 'mL'),  # 1 cm³ = 1 mL
        'l': (1e-6, 'L')      # 1 cm³ = 0.001 L
    }
    
    if unit.lower() in unit_factors:
        return unit_factors[unit.lower()]
    else:
        print(f"Warning: Unknown unit '{unit}'. Using mm³.")
        return unit_factors['mm']

def save_results_to_csv(class_volumes, csv_file, unit):
    """
    Save volume results to CSV file
    
    Args:
        class_volumes (dict): Dictionary with class volumes
        csv_file (str): Path to output CSV file
        unit (str): Volume unit
    """
    try:
        import pandas as pd
        
        # Prepare data for CSV
        csv_data = []
        for class_label, data in class_volumes.items():
            csv_data.append({
                'Class': class_label,
                'Voxel_Count': data['voxel_count'],
                f'Volume_{unit}': data['volume']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
    except ImportError:
        print("\nWarning: pandas not available. CSV export requires pandas.")
        print("Install with: pip install pandas")
    except Exception as e:
        print(f"\nError saving CSV: {e}")

def main():
    """
    Main function to execute volume calculations
    """
    print("NIfTI Volume Calculator")
    print("=" * 40)
    
    # Check if file exists
    if not Path(INPUT_FILE).exists():
        print(f"Error: File '{INPUT_FILE}' not found.")
        print("Please check the INPUT_FILE path in the script configuration.")
        sys.exit(1)
    
    print(f"Loading segmented NIfTI file: {INPUT_FILE}")
    
    # Load NIfTI file
    data, header, affine = load_nifti(INPUT_FILE)
    
    print(f"Image shape: {data.shape}")
    print("data", data)
    print(f"Image data type: {data.dtype}")
    
    # Get voxel spacing
    voxel_spacing = header.get_zooms()[:3]
    print(f"Voxel spacing: {voxel_spacing} mm")
    
    # Get unit conversion factor
    unit_factor, unit_name = get_unit_factor(VOLUME_UNIT)
    
    # Calculate voxel volume
    voxel_volume = get_voxel_volume(header, unit_factor)
    print(f"Voxel volume: {voxel_volume:.6f} {unit_name}")
    
    # Calculate class volumes
    exclude_bg = not INCLUDE_BACKGROUND
    class_volumes = calculate_class_volumes(data, voxel_volume, exclude_bg)
    
    # Check if any classes found
    if not class_volumes:
        print("\nNo classes found in segmentation!")
        if not INCLUDE_BACKGROUND:
            print("Try setting INCLUDE_BACKGROUND = True if you want to include class 0.")
        sys.exit(1)
    
    # Display results
    format_volume_results(class_volumes, unit_name)
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"- Input file: {INPUT_FILE}")
    print(f"- Volume unit: {unit_name}")
    print(f"- Include background: {INCLUDE_BACKGROUND}")
    print(f"- Classes found: {list(class_volumes.keys())}")
    
    # Save to CSV if requested
    if SAVE_CSV:
        save_results_to_csv(class_volumes, SAVE_CSV, VOLUME_UNIT)
    
    print("\nVolume calculation completed successfully!")

if __name__ == "__main__":
    main()