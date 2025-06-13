#!/usr/bin/env python3
"""
Array Volume and Dimension Calculator

This script calculates volumes and dimensions from a 3D segmentation array.
It processes the array and returns measurements for each class.

Requirements:
    - numpy
    - pandas (optional, for CSV export)

Usage:
    Set INPUT_ARRAY and VOXEL_SPACING in configuration
"""

import sys
import time
import numpy as np
import nibabel as nib
from typing import Dict, Tuple, Optional

# Start timing
start_time = time.time()
# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

filepath = "/data/aniket/BrainTumorSegmentation/test_dir/BraTS2021_00234/BraTS2021_00234_seg.nii"
img = nib.load(filepath)  # Load the NIfTI file
data_array = np.array(img.dataobj) # Convert to numpy array

# Array input configuration
INPUT_ARRAY = data_array  # Set this to your numpy array
VOXEL_SPACING = img.header.get_zooms()  # Set this to (x, y, z) spacing in mm

# Volume unit: 'mm', 'cm', 'm', 'ml', 'l'
VOLUME_UNIT = "mm"

# Include background class (0) in calculations
INCLUDE_BACKGROUND = False

# Save results to CSV file (set to None to disable)
SAVE_CSV = None  # e.g., "volume_results.csv"

# ============================================================================

def get_voxel_volume(voxel_spacing: Tuple[float, float, float], unit_factor: float = 1.0) -> float:
    """
    Calculate voxel volume from voxel spacing
    
    Args:
        voxel_spacing: Tuple of (x,y,z) spacing in mm
        unit_factor: Conversion factor (e.g., 1.0 for mm³, 0.001 for cm³)
        
    Returns:
        float: Volume of single voxel in specified units
    """
    # Calculate voxel volume (multiply x, y, z dimensions)
    voxel_volume = np.prod(voxel_spacing)
    
    # Apply unit conversion factor
    voxel_volume *= unit_factor
    
    return voxel_volume

def calculate_class_volumes(segmentation_data: np.ndarray, voxel_volume: float, exclude_background: bool = True) -> Dict:
    """
    Calculate volume for each class in segmentation
    
    Args:
        segmentation_data: 3D segmentation array
        voxel_volume: Volume of single voxel
        exclude_background: Whether to exclude class 0 from results
        
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

def calculate_class_dimensions(segmentation_data: np.ndarray, voxel_spacing: Tuple[float, float, float], 
                             exclude_background: bool = True) -> Dict:
    """
    Calculate height, width, and depth for each class in segmentation
    
    Args:
        segmentation_data: 3D segmentation array
        voxel_spacing: Tuple of (x,y,z) spacing in mm
        exclude_background: Whether to exclude class 0 from results
        
    Returns:
        dict: Dictionary with class labels as keys and dimensions as values
    """
    # Get unique classes in segmentation
    unique_classes = np.unique(segmentation_data)
    
    # Calculate dimensions
    class_dimensions = {}
    
    for class_label in unique_classes:
        # Skip background if requested
        if exclude_background and class_label == 0:
            continue
            
        # Create binary mask for this class
        mask = (segmentation_data == class_label)
        
        # Find non-zero indices
        indices = np.nonzero(mask)
        
        if len(indices[0]) == 0:
            continue
            
        # Calculate bounding box
        min_x, max_x = np.min(indices[0]), np.max(indices[0])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])
        min_z, max_z = np.min(indices[2]), np.max(indices[2])
        
        # Calculate physical dimensions
        height = (max_x - min_x + 1) * voxel_spacing[0]
        width = (max_y - min_y + 1) * voxel_spacing[1]
        depth = (max_z - min_z + 1) * voxel_spacing[2]
        
        class_dimensions[int(class_label)] = {
            'height': height,
            'width': width,
            'depth': depth,
            'voxel_dimensions': {
                'height': max_x - min_x + 1,
                'width': max_y - min_y + 1,
                'depth': max_z - min_z + 1
            }
        }
    
    return class_dimensions

def get_unit_factor(unit: str) -> Tuple[float, str]:
    """
    Get conversion factor for different units
    
    Args:
        unit: Target unit ('mm', 'cm', 'm')
        
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

def calculate_measurements(segmentation_array: np.ndarray, voxel_spacing: Tuple[float, float, float], 
                         unit: str = "mm", include_background: bool = False) -> Dict:
    """
    Calculate all measurements for the segmentation array
    
    Args:
        segmentation_array: 3D segmentation array
        voxel_spacing: Tuple of (x,y,z) spacing in mm
        unit: Volume unit ('mm', 'cm', 'm', 'ml', 'l')
        include_background: Whether to include class 0 in calculations
        
    Returns:
        dict: Dictionary containing all measurements for each class
    """
    # Validate input
    if not isinstance(segmentation_array, np.ndarray):
        raise TypeError("segmentation_array must be a numpy array")
    if len(segmentation_array.shape) != 3:
        raise ValueError("segmentation_array must be a 3D array")
    if len(voxel_spacing) != 3:
        raise ValueError("voxel_spacing must be a tuple of 3 values (x,y,z)")
    
    # Get unit conversion factor
    unit_factor, unit_name = get_unit_factor(unit)
    
    # Calculate voxel volume
    voxel_volume = get_voxel_volume(voxel_spacing, unit_factor)
    
    # Calculate volumes and dimensions
    volumes = calculate_class_volumes(segmentation_array, voxel_volume, not include_background)
    dimensions = calculate_class_dimensions(segmentation_array, voxel_spacing, not include_background)
    
    # Combine results
    results = {}
    for class_label in volumes.keys():
        results[class_label] = {
            'volume': volumes[class_label]['volume'],
            'voxel_count': volumes[class_label]['voxel_count'],
            'height': dimensions[class_label]['height'],
            'width': dimensions[class_label]['width'],
            'depth': dimensions[class_label]['depth'],
            'voxel_dimensions': dimensions[class_label]['voxel_dimensions']
        }
    
    return results

def main():
    """
    Main function to execute calculations
    """
    if INPUT_ARRAY is None:
        print("Error: INPUT_ARRAY must be set")
        sys.exit(1)
    if VOXEL_SPACING is None:
        print("Error: VOXEL_SPACING must be set")
        sys.exit(1)
    
    # Calculate measurements
    results = calculate_measurements(
        INPUT_ARRAY,
        VOXEL_SPACING,
        unit=VOLUME_UNIT,
        include_background=INCLUDE_BACKGROUND
    )
    
    # Print results
    print("\nMeasurement Results:")
    print("=" * 80)
    for class_label, data in sorted(results.items()):
        print(f"\nClass {class_label}:")
        print(f"  Volume: {data['volume']:.2f} {get_unit_factor(VOLUME_UNIT)[1]}")
        print(f"  Voxel Count: {data['voxel_count']}")
        print(f"  Dimensions:")
        print(f"    Height: {data['height']:.2f} mm")
        print(f"    Width: {data['width']:.2f} mm")
        print(f"    Depth: {data['depth']:.2f} mm")
        print(f"  Voxel Dimensions:")
        print(f"    Height: {data['voxel_dimensions']['height']} voxels")
        print(f"    Width: {data['voxel_dimensions']['width']} voxels")
        print(f"    Depth: {data['voxel_dimensions']['depth']} voxels")
    
    # Save to CSV if requested
    if SAVE_CSV:
        try:
            import pandas as pd
            df = pd.DataFrame.from_dict(results, orient='index')
            df.to_csv(SAVE_CSV)
            print(f"\nResults saved to: {SAVE_CSV}")
        except ImportError:
            print("\nWarning: pandas not available. CSV export requires pandas.")
            print("Install with: pip install pandas")

            # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time/60:.2f} minutes")
    
    return results

if __name__ == "__main__":
    main()
