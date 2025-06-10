#!/usr/bin/env python3
"""
NIfTI Volume Calculator

This script calculates volumes and dimensions from either a NIfTI file or a direct array input.
It supports both file-based and array-based processing.

Requirements:
    - nibabel (for file input)
    - numpy
    - pandas (optional, for CSV export)

Usage:
    For file input:
        Set INPUT_FILE in configuration
    For array input:
        Set INPUT_ARRAY and VOXEL_SPACING in configuration
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

# Option 1: File-based input
INPUT_FILE = "/data/aniket/BrainTumorSegmentation/test_dir/BraTS2021_00234/BraTS2021_00234_seg.nii"

# Option 2: Array-based input
INPUT_ARRAY = None  # Set this to your numpy array if using array input
VOXEL_SPACING = None  # Set this to (x, y, z) spacing in mm if using array input

# Volume unit: 'mm', 'cm', 'm', 'ml', 'l'
VOLUME_UNIT = "mm"

# Include background class (0) in calculations
INCLUDE_BACKGROUND = False

# Save results to CSV file (set to None to disable)
SAVE_CSV = None  # e.g., "volume_results.csv"

# ============================================================================

def load_input(input_source: Union[str, np.ndarray], voxel_spacing: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load input data from either a file path or direct array
    
    Args:
        input_source: Either a file path (str) or numpy array
        voxel_spacing: Required if input_source is array, tuple of (x,y,z) spacing in mm
        
    Returns:
        tuple: (data, voxel_spacing)
    """
    if isinstance(input_source, str):
        # File-based input
        try:
            img = nib.load(input_source)
            data = img.get_fdata()
            voxel_spacing = img.header.get_zooms()[:3]
            return data, voxel_spacing
        except Exception as e:
            print(f"Error loading NIfTI file: {e}")
            sys.exit(1)
    elif isinstance(input_source, np.ndarray):
        # Array-based input
        if voxel_spacing is None:
            raise ValueError("voxel_spacing must be provided when using array input")
        if len(voxel_spacing) != 3:
            raise ValueError("voxel_spacing must be a tuple of 3 values (x,y,z)")
        return input_source, voxel_spacing
    else:
        raise TypeError("input_source must be either a file path (str) or numpy array")

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

def calculate_class_dimensions(segmentation_data, voxel_spacing, exclude_background=True):
    """
    Calculate height, width, and depth for each class in segmentation
    
    Args:
        segmentation_data (numpy.ndarray): 3D segmentation array
        voxel_spacing (tuple): Voxel spacing in mm (x, y, z)
        exclude_background (bool): Whether to exclude class 0 from results
        
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

def format_dimension_results(class_dimensions, unit="mm"):
    """
    Format and display dimension results
    
    Args:
        class_dimensions (dict): Dictionary with class dimensions
        unit (str): Unit for dimensions (mm, cm, m)
    """
    print(f"\n{'='*80}")
    print("DIMENSION CALCULATION RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Class':<8} {'Height':<10} {'Width':<10} {'Depth':<10} {'Unit':<8}")
    print("-" * 80)
    
    for class_label, data in sorted(class_dimensions.items()):
        height = data['height']
        width = data['width']
        depth = data['depth']
        
        print(f"{class_label:<8} {height:<10.2f} {width:<10.2f} {depth:<10.2f} {unit:<8}")
    
    print(f"{'='*80}")

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
    
    # Determine input type and load data
    if INPUT_ARRAY is not None:
        print("Using array input")
        if VOXEL_SPACING is None:
            print("Error: VOXEL_SPACING must be provided when using array input")
            sys.exit(1)
        data, voxel_spacing = load_input(INPUT_ARRAY, VOXEL_SPACING)
    else:
        print(f"Loading segmented NIfTI file: {INPUT_FILE}")
        if not Path(INPUT_FILE).exists():
            print(f"Error: File '{INPUT_FILE}' not found.")
            print("Please check the INPUT_FILE path in the script configuration.")
            sys.exit(1)
        data, voxel_spacing = load_input(INPUT_FILE)
    
    print(f"Image shape: {data.shape}")
    print(f"Image data type: {data.dtype}")
    print(f"Voxel spacing: {voxel_spacing} mm")
    
    # Get unit conversion factor
    unit_factor, unit_name = get_unit_factor(VOLUME_UNIT)
    
    # Calculate voxel volume
    voxel_volume = get_voxel_volume(voxel_spacing, unit_factor)
    print(f"Voxel volume: {voxel_volume:.6f} {unit_name}")
    
    # Calculate class volumes
    exclude_bg = not INCLUDE_BACKGROUND
    class_volumes = calculate_class_volumes(data, voxel_volume, exclude_bg)
    
    # Calculate class dimensions
    class_dimensions = calculate_class_dimensions(data, voxel_spacing, exclude_bg)
    
    # Check if any classes found
    if not class_volumes:
        print("\nNo classes found in segmentation!")
        if not INCLUDE_BACKGROUND:
            print("Try setting INCLUDE_BACKGROUND = True if you want to include class 0.")
        sys.exit(1)
    
    # Display volume results
    format_volume_results(class_volumes, unit_name)
    
    # Display dimension results
    format_dimension_results(class_dimensions)
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    if INPUT_ARRAY is not None:
        print(f"- Input type: Array")
        print(f"- Array shape: {INPUT_ARRAY.shape}")
    else:
        print(f"- Input type: File")
        print(f"- Input file: {INPUT_FILE}")
    print(f"- Volume unit: {unit_name}")
    print(f"- Include background: {INCLUDE_BACKGROUND}")
    print(f"- Classes found: {list(class_volumes.keys())}")
    
    # Save to CSV if requested
    if SAVE_CSV:
        save_results_to_csv(class_volumes, SAVE_CSV, VOLUME_UNIT)
    
    print("\nVolume and dimension calculation completed successfully!")

if __name__ == "__main__":
    main()