# Brain Tumor Segmentation Test Script

This repository contains a comprehensive test script for brain tumor segmentation using the BraTS 2021 challenge methodology. The script processes NIfTI images and performs automated brain tumor segmentation using trained deep learning models.

## Overview

The test script implements the complete pipeline described in the research paper:
1. **Preprocessing**: Z-score normalization and cropping to 128×128×128
2. **Binary Segmentation**: ROI detection using binary model
3. **ROI Cropping**: Smart cropping based on binary mask 
4. **Multiclass Segmentation**: Detailed segmentation using attention UNet
5. **Postprocessing**: Converting predictions back to standard format
6. **NIfTI Output**: Saving results in medical imaging format

## Quick Start

### 1. Test Setup
First, run the setup test to verify all dependencies:

```bash
python test_setup.py
```

This will check:
- ✅ All required Python packages
- ✅ GPU availability (optional)
- ✅ File structure and test data
- ✅ Model definitions
- ✅ Preprocessing functions

### 2. Run Inference
If setup tests pass, run the main inference script:

```bash
python test_nifti_inference.py
```

### 3. Advanced Usage
```bash
# Custom directories
python test_nifti_inference.py --test_dir my_test_data --output_dir my_results

# Skip intermediate files to save space
python test_nifti_inference.py --no_intermediate

# Custom model weights
python test_nifti_inference.py --multiclass_weights path/to/weights.h5
```

## File Structure

Your directory should look like this:

```
BrainTumorSegmentation/
├── test_nifti_inference.py    # Main inference script
├── test_setup.py              # Setup verification script
├── utils/                     # Utility functions
│   ├── models.py             # Model architectures
│   ├── preprocessing.py      # Data preprocessing
│   └── ...
├── weights/                   # Trained model weights
│   └── weights_1000instances.h5
├── test_dir/                  # Test data directory
│   ├── BraTS2021_00000/      # Patient directory
│   │   ├── BraTS2021_00000_t1.nii
│   │   ├── BraTS2021_00000_t1ce.nii
│   │   ├── BraTS2021_00000_t2.nii
│   │   ├── BraTS2021_00000_flair.nii
│   │   └── BraTS2021_00000_seg.nii (ground truth)
│   └── BraTS2021_00002/      # Another patient
└── test_results/              # Output directory (created automatically)
    ├── predictions/           # Final segmentation results
    ├── intermediate/          # Step-by-step results
    └── logs/                 # Processing reports
```

## Input Data Format

The script expects BraTS-format data:
- **Patient Directory**: Named like `BraTS2021_XXXXX`
- **MRI Modalities**: 
  - `{patient_id}_t1.nii` - T1-weighted
  - `{patient_id}_t1ce.nii` - T1-weighted with contrast
  - `{patient_id}_t2.nii` - T2-weighted  
  - `{patient_id}_flair.nii` - FLAIR
- **Optional Ground Truth**: `{patient_id}_seg.nii`

## Output Files

### Final Predictions
- **Location**: `test_results/predictions/`
- **Format**: `{patient_id}_prediction.nii.gz`
- **Labels**: 
  - 0 = Background
  - 1 = Necrotic Core (NCR)
  - 2 = Edema (ED)
  - 4 = Enhancing Tumor (ET)

### Intermediate Results (Optional)
- **Location**: `test_results/intermediate/`
- **Files**:
  - `{patient_id}_normalized.nii.gz` - Preprocessed MRI
  - `{patient_id}_binary_mask.nii.gz` - Binary ROI mask
  - `{patient_id}_binary_prob.nii.gz` - Binary probabilities
  - `{patient_id}_cropped.nii.gz` - Cropped MRI data
  - `{patient_id}_multiclass_raw.nii.gz` - Raw multiclass output
  - `{patient_id}_prob_class_X.nii.gz` - Per-class probabilities

### Processing Reports
- **Location**: `test_results/logs/`
- **Files**:
  - `{patient_id}_report.json` - Detailed statistics
  - `inference_summary.json` - Overall summary
  - `test_inference.log` - Processing log

## Model Architecture

### Binary Model
- **Purpose**: ROI detection and cropping
- **Architecture**: 3D U-Net with instance normalization
- **Input**: 128×128×128×4 (4 MRI modalities)
- **Output**: 128×128×128×1 (binary mask)

### Multiclass Model  
- **Purpose**: Detailed tumor segmentation
- **Architecture**: 3D Attention U-Net with channel attention
- **Input**: 48×48×128×5 (cropped MRI + binary mask)
- **Output**: 48×48×128×4 (4 segmentation classes)

## BraTS Label System

The script follows the BraTS 2021 challenge labeling convention:

| Label | Structure | Description |
|-------|-----------|-------------|
| 0 | Background | Normal brain tissue |
| 1 | NCR | Necrotic Core |
| 2 | ED | Edema/Invasion |
| 4 | ET | Enhancing Tumor |

**Hierarchical Regions:**
- **Whole Tumor (WT)**: Labels 1+2+4 (all tumor regions)
- **Tumor Core (TC)**: Labels 1+4 (solid tumor)
- **Enhancing Tumor (ET)**: Label 4 only

## Preprocessing Pipeline

1. **Loading**: Read NIfTI files for all 4 modalities
2. **Cropping**: Center crop to 128×128×128 from 240×240×155
3. **Normalization**: Z-score normalization per modality
4. **Stacking**: Combine modalities into 4D tensor

## Processing Pipeline

1. **Binary ROI Detection**:
   - Input: 128×128×128×4 normalized MRI
   - Output: Binary mask indicating tumor presence
   
2. **Smart Cropping**:
   - Use binary mask to find tumor region
   - Crop to 48×48×128 around tumor center
   - Add padding if tumor is too small
   
3. **Multiclass Segmentation**:
   - Input: Cropped MRI + binary mask (48×48×128×5)
   - Output: Detailed segmentation (48×48×128×4)
   
4. **Postprocessing**:
   - Map predictions back to 128×128×128 space
   - Convert class probabilities to discrete labels
   - Apply BraTS label convention

## Performance Notes

- **Processing Time**: ~30-60 seconds per patient (GPU)
- **Memory Usage**: ~4-8GB GPU memory
- **CPU Alternative**: Works on CPU (slower: ~5-10 minutes per patient)

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing Model Weights**:
   - Download trained weights or train your own models
   - Update paths in command line arguments

3. **CUDA/GPU Issues**:
   - Script will fall back to CPU automatically
   - Check GPU memory if encountering OOM errors

4. **File Format Issues**:
   - Ensure NIfTI files are properly formatted
   - Check patient directory naming convention

### Debug Mode

Enable detailed logging:
```bash
python test_nifti_inference.py --verbose
```

View intermediate results to debug pipeline:
```bash
# All intermediate files will be saved
python test_nifti_inference.py  

# Skip intermediate files for speed
python test_nifti_inference.py --no_intermediate
```

## Research Reference

This implementation is based on the paper:
> "Brain Tumor Segmentation with Region of Interest Detection and Attention Mechanisms"

Key features implemented:
- ✅ ROI-based preprocessing to reduce computation
- ✅ Channel-wise attention mechanisms  
- ✅ Test-time augmentation capability
- ✅ Uncertainty estimation through multiple predictions
- ✅ BraTS 2021 evaluation metrics

## Example Usage

```python
# Programmatic usage
from test_nifti_inference import BrainTumorSegmentationTester

tester = BrainTumorSegmentationTester(
    test_dir='my_data',
    output_dir='my_results'
)

tester.run_inference()
```

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- nibabel
- numpy  
- tensorflow-addons
- pathlib
- argparse

Install all at once:
```bash
pip install -r requirements.txt
```

---

**For questions or issues, please check the logs in `test_results/logs/` for detailed error messages.** 