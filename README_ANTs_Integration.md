# Brain Tumor Segmentation with ANTs Resampling

## Overview

The brain tumor segmentation test script (`test_nifti_inference.py`) has been enhanced with ANTs (Advanced Normalization Tools) integration for proper spatial resampling of both binary and multiclass model predictions to ground truth reference space.

## Key Features

### ANTs Integration
- **Automatic Reference Detection**: Uses ground truth segmentation (`_seg.nii`) as reference when available, falls back to T1 image
- **Proper Spatial Alignment**: Maintains correct affine transformations and image geometry
- **Nearest Neighbor Interpolation**: Preserves segmentation label integrity during resampling
- **Dual Output**: Saves both original processing space and resampled predictions

### Enhanced Pipeline
1. **Binary Model Prediction**: ROI detection in 128³ space
2. **Multiclass Model Prediction**: Attention-based segmentation in 48³ space  
3. **Postprocessing**: Reconstruction to original data dimensions
4. **ANTs Resampling**: Spatial alignment to ground truth reference space
5. **Comprehensive Reporting**: Statistics for both original and resampled predictions

## Installation Requirements

```bash
pip install antspyx
```

## Usage

### Basic Usage
```bash
python test_nifti_inference.py
```

### Advanced Options
```bash
python test_nifti_inference.py \
    --test_dir test_dir \
    --output_dir test_results \
    --binary_weights path/to/binary/weights.hdf5 \
    --multiclass_weights path/to/multiclass/weights.h5 \
    --no_intermediate \
    --no_resampling
```

### Command Line Arguments
- `--test_dir`: Directory containing patient folders with BraTS format data
- `--output_dir`: Directory to save results (default: `test_results`)
- `--binary_weights`: Path to binary model weights
- `--multiclass_weights`: Path to multiclass model weights  
- `--no_intermediate`: Skip saving intermediate processing results
- `--no_resampling`: Disable ANTs resampling (process in original space only)

## Expected Data Structure

```
test_dir/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii
│   ├── BraTS2021_00000_t1ce.nii
│   ├── BraTS2021_00000_t2.nii
│   ├── BraTS2021_00000_flair.nii
│   └── BraTS2021_00000_seg.nii     # Ground truth (optional)
└── BraTS2021_00002/
    ├── BraTS2021_00002_t1.nii
    ├── BraTS2021_00002_t1ce.nii
    ├── BraTS2021_00002_t2.nii
    ├── BraTS2021_00002_flair.nii
    └── BraTS2021_00002_seg.nii     # Ground truth (optional)
```

## Output Structure

```
test_results/
├── predictions/                    # Original processing space
│   ├── BraTS2021_00000_prediction.nii.gz
│   └── BraTS2021_00002_prediction.nii.gz
├── predictions_resampled/          # ANTs resampled to reference space
│   ├── BraTS2021_00000_binary_resampled.nii.gz
│   ├── BraTS2021_00000_multiclass_resampled.nii.gz
│   ├── BraTS2021_00002_binary_resampled.nii.gz
│   └── BraTS2021_00002_multiclass_resampled.nii.gz
├── intermediate/                   # Debug and analysis files
│   ├── *_normalized.nii.gz
│   ├── *_binary_mask.nii.gz
│   ├── *_binary_prob.nii.gz
│   ├── *_cropped.nii.gz
│   ├── *_multiclass_raw.nii.gz
│   └── *_prob_class_*.nii.gz
├── logs/                          # Detailed patient reports
│   ├── BraTS2021_00000_report.json
│   └── BraTS2021_00002_report.json
└── inference_summary.json         # Overall processing summary
```

## Resampling Process

### 1. Reference Selection
- **Priority 1**: Ground truth segmentation (`_seg.nii`) if available
- **Priority 2**: T1 anatomical image (`_t1.nii`) as fallback

### 2. Spatial Alignment
- Creates ANTs image objects from numpy predictions
- Sets geometry (spacing, origin, direction) to match reference
- Performs resampling using `ants.resample_image_to_target()`
- Uses nearest neighbor interpolation to preserve label integrity

### 3. Output Generation
- **Binary Mask**: Resampled ROI detection results
- **Multiclass Segmentation**: Resampled final tumor segmentation
- **Both outputs**: Properly aligned to ground truth space with correct affine transformations

## Report Structure

### Patient Report Example
```json
{
  "patient_id": "BraTS2021_00002",
  "processing_time_seconds": 6.17,
  "original_segmentation": {
    "shape": [240, 240, 155],
    "total_voxels": 8928000,
    "label_statistics": {
      "0": {"name": "Background", "voxel_count": 8907531, "percentage": 99.77},
      "1": {"name": "Necrotic Core (NCR)", "voxel_count": 592, "percentage": 0.01},
      "2": {"name": "Edema (ED)", "voxel_count": 17521, "percentage": 0.20},
      "4": {"name": "Enhancing Tumor (ET)", "voxel_count": 2356, "percentage": 0.03}
    }
  },
  "resampling": {
    "reference_used": "ground_truth",
    "binary_mask": {
      "shape": [240, 240, 155],
      "tumor_voxels": 126724,
      "tumor_percentage": 1.42
    },
    "multiclass_segmentation": {
      "shape": [240, 240, 155],
      "label_statistics": { /* Same structure as original */ }
    }
  }
}
```

## Key Benefits

### Spatial Accuracy
- **Proper Alignment**: Predictions aligned to ground truth coordinate system
- **Preserved Geometry**: Maintains correct voxel spacing and orientation
- **Medical Compliance**: Output compatible with standard medical imaging tools

### Analysis Ready
- **Ground Truth Comparison**: Direct comparison possible with reference segmentations
- **Evaluation Metrics**: Ready for dice score, Hausdorff distance calculations
- **Visualization**: Proper overlay with original images in medical viewers

### Debugging Support
- **Intermediate Results**: Full pipeline visibility for troubleshooting
- **Processing Statistics**: Detailed quantitative analysis at each stage
- **Comprehensive Logging**: Step-by-step processing information

## Technical Details

### Model Pipeline
1. **Preprocessing**: Z-score normalization, center cropping to 128³
2. **Binary Model**: U-Net architecture for ROI detection
3. **ROI Cropping**: Tumor-focused 48³ patches extraction
4. **Multiclass Model**: Attention U-Net for final segmentation
5. **Postprocessing**: Reconstruction to original dimensions
6. **ANTs Resampling**: Spatial alignment to reference space

### Label Format
- **Background**: 0
- **Necrotic Core (NCR)**: 1  
- **Edema (ED)**: 2
- **Enhancing Tumor (ET)**: 4

### Performance
- **Processing Time**: ~4-6 seconds per patient (including resampling)
- **Memory Usage**: Optimized for standard workstation hardware
- **GPU Support**: Automatic GPU utilization when available

## Troubleshooting

### Common Issues
1. **Missing Ground Truth**: Falls back to T1 reference automatically
2. **Shape Mismatches**: ANTs handles different image dimensions automatically  
3. **Memory Issues**: Consider using `--no_intermediate` flag
4. **ANTs Installation**: Ensure `antspyx` is installed correctly

### Validation
Run the script on test data to verify:
- Both original and resampled outputs are generated
- Image dimensions match reference space (240×240×155 for BraTS)
- Label statistics are preserved after resampling
- Processing completes without errors

## Citation

If you use this enhanced segmentation pipeline, please cite the original paper:
- DOI: 10.1007/s11042-024-20443-0
- BrainTumorSegmentation with attention-based models and ANTs integration 