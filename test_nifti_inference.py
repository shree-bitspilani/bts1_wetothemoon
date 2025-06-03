#!/usr/bin/env python3
"""
Brain Tumor Segmentation Test Script
====================================

This script processes NIfTI images from test_dir and performs brain tumor segmentation
using the trained binary and multiclass models. It saves predictions and intermediate
results in NIfTI format for analysis. Uses ANTs for proper resampling to ground truth space.

Usage:
    python test_nifti_inference.py

Requirements:
    - Trained models in weights/ directory
    - Test data in test_dir/
    - All dependencies from requirements.txt
    - antspyx for resampling
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import tensorflow as tf
from keras.layers import ELU
import argparse
from datetime import datetime
import json
import logging
import ants

# Add utils to path
sys.path.append('utils')

from utils.models import binary_model, attention_multiclass_model
from utils.preprocessing import normalize_mri_data, roi_crop, calc_z_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BrainTumorSegmentationTester:
    """Main class for brain tumor segmentation testing with ANTs resampling."""
    
    def __init__(self, 
                 test_dir='/data/aniket/BrainTumorSegmentation/test_dir',
                 output_dir='/data/aniket/BrainTumorSegmentation/test_results',
                 binary_weights='/data/aniket/BrainTumorSegmentation/weights/BinaryWeights.hdf5',
                 multiclass_weights='/data/aniket/BrainTumorSegmentation/weights/weights_1000instances.h5',
                 save_intermediate=True,
                 use_ground_truth_space=True,
                 use_training_style_preprocessing=True):
        
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.binary_weights = binary_weights
        self.multiclass_weights = multiclass_weights
        self.save_intermediate = save_intermediate
        self.use_ground_truth_space = use_ground_truth_space
        self.use_training_style_preprocessing = use_training_style_preprocessing
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'predictions_resampled').mkdir(exist_ok=True)
        (self.output_dir / 'intermediate').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize models
        self.binary_model = None
        self.multiclass_model = None
        
        # BraTS label mapping
        self.label_mapping = {
            0: 'Background',
            1: 'Necrotic Core (NCR)',
            2: 'Edema (ED)', 
            3: 'Enhancing Tumor (ET)',
            4: 'Enhancing Tumor (ET)'  # Original ET label in BraTS
        }
        
        logger.info(f"Initialized tester with test_dir: {self.test_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use ground truth space for resampling: {self.use_ground_truth_space}")
        logger.info(f"Use training-style preprocessing: {self.use_training_style_preprocessing}")
    
    def load_models(self):
        """Load the trained binary and multiclass models."""
        logger.info("Loading models...")
        
        # Load binary model
        try:
            n_channels = 20
            self.binary_model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())
            
            # Check if binary weights exist, if not use placeholder
            if os.path.exists(self.binary_weights):
                self.binary_model.load_weights(self.binary_weights)
                logger.info(f"Loaded binary model weights from {self.binary_weights}")
            else:
                logger.warning(f"Binary weights not found at {self.binary_weights}. Using random weights.")
                
        except Exception as e:
            logger.error(f"Error loading binary model: {e}")
            raise
        
        # Load multiclass model
        try:
            self.multiclass_model = attention_multiclass_model(48, 48, 128, 4, 4)
            
            if os.path.exists(self.multiclass_weights):
                self.multiclass_model.load_weights(self.multiclass_weights)
                logger.info(f"Loaded multiclass model weights from {self.multiclass_weights}")
            else:
                logger.warning(f"Multiclass weights not found at {self.multiclass_weights}. Using random weights.")
                
        except Exception as e:
            logger.error(f"Error loading multiclass model: {e}")
            raise
    
    def load_patient_data(self, patient_dir):
        """Load all MRI modalities and ground truth for a patient."""
        patient_path = Path(patient_dir)
        patient_id = patient_path.name
        
        logger.info(f"Loading data for patient: {patient_id}")
        
        # Expected file patterns for BraTS data
        modalities = {
            't1': f"{patient_id}_t1.nii",
            't1ce': f"{patient_id}_t1ce.nii", 
            't2': f"{patient_id}_t2.nii",
            'flair': f"{patient_id}_flair.nii",
            'seg': f"{patient_id}_seg.nii"  # Ground truth segmentation
        }
        
        loaded_data = {}
        headers = {}
        ants_images = {}
        
        for modality, filename in modalities.items():
            filepath = patient_path / filename
            
            if not filepath.exists():
                if modality == 'seg':
                    logger.warning(f"Ground truth segmentation not found: {filepath}")
                    continue
                else:
                    logger.error(f"Missing file: {filepath}")
                    raise FileNotFoundError(f"Required file not found: {filepath}")
            
            try:
                # Load with nibabel
                nii_img = nib.load(filepath)
                loaded_data[modality] = nii_img.get_fdata()
                headers[modality] = nii_img.header
                
                # Load with ANTs for resampling
                ants_images[modality] = ants.image_read(str(filepath))
                
                logger.info(f"Loaded {modality}: {loaded_data[modality].shape}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                raise
        
        return loaded_data, headers, ants_images, patient_id
    
    def resample_to_reference(self, prediction_array, reference_ants_image, prediction_name="prediction"):
        """Resample prediction to match reference image using ANTs."""
        logger.info(f"Resampling {prediction_name} to reference space using ANTs")
        
        try:
            # Create ANTs image from prediction array
            prediction_ants = ants.from_numpy(prediction_array.astype(np.float32))
            
            # Get the reference image geometry
            ref_spacing = reference_ants_image.spacing
            ref_origin = reference_ants_image.origin
            ref_direction = reference_ants_image.direction
            
            # Set the prediction image geometry to match our processing space
            # This is important for proper spatial alignment
            prediction_ants.set_spacing(ref_spacing)
            prediction_ants.set_origin(ref_origin)
            prediction_ants.set_direction(ref_direction)
            
            # Resample using ANTs with nearest neighbor interpolation for segmentation
            resampled_ants = ants.resample_image_to_target(
                image=prediction_ants,
                target=reference_ants_image,
                interp_type='nearestNeighbor'  # Use nearest neighbor for segmentation labels
            )
            
            # Convert back to numpy array
            resampled_array = resampled_ants.numpy()
            
            logger.info(f"Original {prediction_name} shape: {prediction_array.shape}")
            logger.info(f"Resampled {prediction_name} shape: {resampled_array.shape}")
            logger.info(f"Reference shape: {reference_ants_image.shape}")
            
            return resampled_array, resampled_ants
            
        except Exception as e:
            logger.error(f"Error resampling {prediction_name}: {e}")
            raise

    def save_ants_image(self, ants_image, filepath):
        """Save ANTs image to file."""
        try:
            ants.image_write(ants_image, str(filepath))
            logger.info(f"Saved ANTs image: {filepath}")
        except Exception as e:
            logger.error(f"Error saving ANTs image {filepath}: {e}")
            raise

    def calculate_dice_coefficient(self, y_true, y_pred):
        """
        Calculate Dice coefficient for binary segmentation using Keras backend operations.
        Based on metric.py implementation.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            dice_score: Dice coefficient (0-1, higher is better)
        """
        # Ensure binary masks
        y_true = (y_true > 0).astype(np.float32)
        y_pred = (y_pred > 0).astype(np.float32)
        
        # Flatten arrays
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        
        # Calculate intersection and sums
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f)
        
        if union == 0:
            # Both masks are empty, perfect match
            return 1.0
        
        # Calculate dice with smooth=1 as in metric.py
        smooth = 1
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)

    def calculate_multiclass_dice(self, y_true, y_pred, classes=None):
        """
        Calculate Dice coefficient for each class in multiclass segmentation.
        Based on metric.py implementation for multilabel dice.
        
        Args:
            y_true: Ground truth segmentation with class labels
            y_pred: Predicted segmentation with class labels  
            classes: List of class labels to evaluate (if None, uses unique values)
            
        Returns:
            dice_scores: Dictionary with Dice scores for each class and averages
        """
        if classes is None:
            # Get unique classes from both ground truth and prediction
            classes = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
        
        dice_scores = {}
        
        # Convert to one-hot encoding for each class
        for class_id in classes:
            if class_id == 0:  # Skip background
                continue
            
            # Create binary masks for current class
            true_class = (y_true == class_id).astype(np.float32)
            pred_class = (y_pred == class_id).astype(np.float32)
            
            # Calculate dice using Keras-style implementation
            true_f = true_class.flatten()
            pred_f = pred_class.flatten()
            
            intersection = np.sum(true_f * pred_f)
            union = np.sum(true_f) + np.sum(pred_f)
            
            smooth = 1
            dice_score = (2.0 * intersection + smooth) / (union + smooth) if union > 0 else 1.0
            
            class_name = self.label_mapping.get(int(class_id), f"Class_{class_id}")
            dice_scores[int(class_id)] = {
                'name': class_name,
                'dice_score': round(dice_score, 4)
            }
        
        # Calculate mean Dice excluding background (class 0)
        non_bg_dice = [dice_scores[i]['dice_score'] for i in dice_scores if i > 0]
        mean_dice = np.mean(non_bg_dice) if non_bg_dice else 0.0
        
        # Calculate overall mean including background
        all_dice = [dice_scores[i]['dice_score'] for i in dice_scores]
        mean_dice_with_bg = np.mean(all_dice) if all_dice else 0.0
        
        dice_scores['mean_dice'] = round(mean_dice, 4)
        dice_scores['mean_dice_with_bg'] = round(mean_dice_with_bg, 4)
        
        return dice_scores

    def evaluate_predictions(self, binary_pred, multiclass_pred, ground_truth, patient_id):
        """
        Evaluate predictions against ground truth using Dice coefficient.
        
        Args:
            binary_pred: Binary prediction mask
            multiclass_pred: Multiclass prediction segmentation
            ground_truth: Ground truth segmentation
            patient_id: Patient identifier for logging
            
        Returns:
            evaluation_results: Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating predictions for {patient_id}")
        
        evaluation_results = {
            'has_ground_truth': True,
            'binary_evaluation': {},
            'multiclass_evaluation': {}
        }
        
        try:
            # Ensure all arrays have the same shape for comparison
            if binary_pred.shape != ground_truth.shape:
                logger.warning(f"Shape mismatch - Binary pred: {binary_pred.shape}, GT: {ground_truth.shape}")
                # Try to handle shape mismatch by cropping or padding
                min_shape = tuple(min(a, b) for a, b in zip(binary_pred.shape, ground_truth.shape))
                binary_pred = binary_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
                ground_truth_binary = ground_truth[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                ground_truth_binary = ground_truth.copy()
                
            if multiclass_pred.shape != ground_truth.shape:
                logger.warning(f"Shape mismatch - Multiclass pred: {multiclass_pred.shape}, GT: {ground_truth.shape}")
                min_shape = tuple(min(a, b) for a, b in zip(multiclass_pred.shape, ground_truth.shape))
                multiclass_pred_eval = multiclass_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
                ground_truth_multiclass = ground_truth[:min_shape[0], :min_shape[1], :min_shape[2]]
            else:
                multiclass_pred_eval = multiclass_pred.copy()
                ground_truth_multiclass = ground_truth.copy()

            # Binary evaluation: compare any tumor (GT > 0) vs binary prediction
            gt_binary = (ground_truth_binary > 0).astype(np.uint8)
            binary_dice = self.calculate_dice_coefficient(gt_binary, binary_pred)
            
            evaluation_results['binary_evaluation'] = {
                'dice_score': round(binary_dice, 4),
                'ground_truth_tumor_voxels': int(np.sum(gt_binary)),
                'predicted_tumor_voxels': int(np.sum(binary_pred > 0)),
                'total_voxels': int(gt_binary.size)
            }
            
            logger.info(f"Binary Dice score: {binary_dice:.4f}")
            
            # Multiclass evaluation: compare each class
            multiclass_dice = self.calculate_multiclass_dice(ground_truth_multiclass, multiclass_pred_eval)
            evaluation_results['multiclass_evaluation'] = multiclass_dice
            
            logger.info(f"Multiclass mean Dice (excluding background): {multiclass_dice['mean_dice']:.4f}")
            
            # Log individual class scores
            for class_id, class_info in multiclass_dice.items():
                if isinstance(class_info, dict) and 'dice_score' in class_info:
                    logger.info(f"  {class_info['name']}: {class_info['dice_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in evaluation for {patient_id}: {e}")
            evaluation_results['evaluation_error'] = str(e)
        
        return evaluation_results

    def calc_z_score_training_style(self, img):
        """
        Calculate z-score normalization exactly as done in training.
        Based on utils.preprocessing.calc_z_score()
        """
        img_normalized = img.copy().astype(np.float32)
        
        # Only normalize non-zero voxels
        non_zero_mask = img_normalized != 0
        if np.sum(non_zero_mask) > 0:
            avg_pixel_value = np.sum(img_normalized) / np.count_nonzero(img_normalized)
            sd_pixel_value = np.std(img_normalized[non_zero_mask])
            
            if sd_pixel_value > 0:
                for i in range(img_normalized.shape[0]):
                    for j in range(img_normalized.shape[1]):
                        for k in range(img_normalized.shape[2]):
                            if img_normalized[i, j, k] != 0:
                                img_normalized[i, j, k] = (img_normalized[i, j, k] - avg_pixel_value) / sd_pixel_value
        
        return img_normalized

    def preprocess_data_training_style(self, data_dict, patient_id):
        """
        Preprocess MRI data exactly as done during training.
        Based on utils.preprocessing.normalize_mri_data()
        
        This function:
        1. Crops from 240x240x155 to 128x128x128 using exact training indices
        2. Applies z-score normalization per modality 
        3. Stacks in training order: [flair, t1ce, t1, t2]
        4. Handles mask preprocessing for ground truth
        """
        logger.info(f"Preprocessing data for {patient_id} using TRAINING-STYLE preprocessing")
        
        t1 = data_dict['t1']
        t1ce = data_dict['t1ce'] 
        t2 = data_dict['t2']
        flair = data_dict['flair']
        
        logger.info(f"Original image shapes: T1={t1.shape}, T1CE={t1ce.shape}, T2={t2.shape}, FLAIR={flair.shape}")
        
        # CRITICAL: Use exact same cropping as training [56:184, 56:184, 13:141]
        # This crops from (240, 240, 155) to (128, 128, 128)
        crop_indices = (slice(56, 184), slice(56, 184), slice(13, 141))
        
        try:
            # Crop each modality to training size
            t2_cropped = t2[crop_indices]
            t1ce_cropped = t1ce[crop_indices] 
            flair_cropped = flair[crop_indices]
            t1_cropped = t1[crop_indices]
            
            logger.info(f"Cropped shapes: T1={t1_cropped.shape}, T1CE={t1ce_cropped.shape}, T2={t2_cropped.shape}, FLAIR={flair_cropped.shape}")
            
            # Apply z-score normalization per modality (training style)
            t2_norm = self.calc_z_score_training_style(t2_cropped)
            t1ce_norm = self.calc_z_score_training_style(t1ce_cropped)
            flair_norm = self.calc_z_score_training_style(flair_cropped)
            t1_norm = self.calc_z_score_training_style(t1_cropped)
            
            # Stack in training order: [flair, t1ce, t1, t2]
            normalized_data = np.stack([flair_norm, t1ce_norm, t1_norm, t2_norm], axis=3)
            
            logger.info(f"Final normalized data shape: {normalized_data.shape}")
            
            # Process ground truth mask if available
            processed_mask = None
            if 'seg' in data_dict:
                mask = data_dict['seg']
                mask_cropped = mask[crop_indices].astype(np.uint8)
                
                # Convert label 4 to 3 (training style)
                mask_cropped[mask_cropped == 4] = 3
                
                # Create one-hot encoding (training style)
                processed_mask = self.change_mask_shape_training_style(mask_cropped)
                logger.info(f"Processed mask shape: {processed_mask.shape}")
            else:
                # Create dummy mask for compatibility
                processed_mask = np.zeros((128, 128, 128, 4))
            
            # Save intermediate results if requested
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    normalized_data, 
                    f"{patient_id}_normalized_training_style.nii.gz"
                )
                
                if 'seg' in data_dict:
                    self.save_intermediate_nifti(
                        mask_cropped,
                        f"{patient_id}_mask_cropped.nii.gz"
                    )
            
            return normalized_data, processed_mask, crop_indices
            
        except Exception as e:
            logger.error(f"Error in training-style preprocessing: {e}")
            raise

    def change_mask_shape_training_style(self, mask):
        """
        Convert mask to one-hot encoding exactly as done in training.
        Based on utils.preprocessing.change_mask_shape()
        """
        if mask.shape == (128, 128, 128, 4):
            logger.warning("Mask shape is already (128, 128, 128, 4)")
            return mask

        new_mask = np.zeros((128, 128, 128, 4))
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    label = int(mask[i, j, k])
                    if 0 <= label < 4:
                        new_mask[i, j, k, label] = 1

        return new_mask

    def predict_binary_mask_training_style(self, normalized_data, patient_id):
        """
        Generate binary mask using training-style preprocessing.
        Input data is already in 128x128x128 format from training-style preprocessing.
        """
        logger.info(f"Generating binary mask for {patient_id} using training-style data")
        
        try:
            # Data is already 128x128x128x4 from training-style preprocessing
            logger.info(f"Input data shape: {normalized_data.shape}")
            
            if normalized_data.shape[:3] != (128, 128, 128):
                logger.error(f"Expected 128x128x128 input, got {normalized_data.shape[:3]}")
                raise ValueError("Input data must be 128x128x128 for training-style processing")
            
            # Prepare input for binary model
            img_input = np.expand_dims(normalized_data, axis=0)
            
            # Predict using binary model
            binary_pred = self.binary_model.predict(img_input, verbose=0)
            binary_mask = binary_pred[0, :, :, :, 0]
            
            # Threshold to get binary mask
            binary_mask_thresh = (binary_mask > 0.5).astype(np.uint8)
            
            logger.info(f"Binary mask shape: {binary_mask_thresh.shape}")
            logger.info(f"Binary mask unique values: {np.unique(binary_mask_thresh)}")
            
            # Save intermediate results
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    binary_mask_thresh,
                    f"{patient_id}_binary_mask_training_style.nii.gz"
                )
                
                self.save_intermediate_nifti(
                    binary_mask,
                    f"{patient_id}_binary_prob_training_style.nii.gz"
                )
            
            return binary_mask_thresh
            
        except Exception as e:
            logger.error(f"Error in training-style binary prediction: {e}")
            raise

    def crop_with_binary_mask_training_style(self, normalized_data, mask, binary_mask, patient_id):
        """
        Crop data using binary mask exactly as done during training.
        Based on utils.preprocessing.roi_crop() and global_extraction()
        """
        logger.info(f"Cropping data using training-style ROI detection for {patient_id}")
        
        try:
            # Use binary mask to find tumor location (training style)
            binary_mask_expanded = np.expand_dims(binary_mask, axis=-1)
            loc = np.where(binary_mask == 1)
            
            if len(loc[0]) == 0:
                # No tumor detected, use center crop as fallback
                logger.warning(f"No tumor region detected for {patient_id}, using center crop")
                a, b = 40, 88  # Center crop for 48x48
                c, d = 40, 88
            else:
                # ROI crop based on tumor location (training style)
                thresh = 12
                a = max(0, np.amin(loc[0]) - thresh)
                b = min(128, np.amax(loc[0]) + thresh)
                c = max(0, np.amin(loc[1]) - thresh)
                d = min(128, np.amax(loc[1]) + thresh)

                # Ensure minimum size of 48x48 (training requirement)
                while abs(b - a) < 48:
                    a = max(0, a - 1)
                    b = min(128, b + 1)

                while abs(d - c) < 48:
                    c = max(0, c - 1)
                    d = min(128, d + 1)
            
            # Store cropping coordinates for reconstruction
            crop_coords = {
                'h_start': a, 'h_end': b,
                'w_start': c, 'w_end': d,
                'd_start': 0, 'd_end': 128,
                'original_shape': (128, 128, 128),  # Training uses 128x128x128 space
                'training_style': True
            }
            
            # Crop to ROI and add binary mask as 5th channel (training style)
            cropped_img = normalized_data[a:b, c:d, :]  # 4 channels
            cropped_binary = binary_mask[a:b, c:d, :]
            
            # Concatenate binary mask as 5th channel
            img_with_binary = np.concatenate([cropped_img, binary_mask_expanded[a:b, c:d, :]], axis=-1)
            
            # Crop mask for compatibility
            cropped_mask = mask[a:b, c:d, :]
            
            logger.info(f"ROI crop coordinates: [{a}:{b}, {c}:{d}, 0:128]")
            logger.info(f"Cropped image with binary shape: {img_with_binary.shape}")
            logger.info(f"Cropped mask shape: {cropped_mask.shape}")
            
            # For multiclass training, we need exactly 48x48x128x4 (without binary mask channel)
            # This matches the training data preparation
            final_cropped_img = cropped_img  # Use only the 4 MRI channels
            
            # Ensure exact 48x48x128 size by cropping or padding
            target_h, target_w, target_d = 48, 48, 128
            current_h, current_w, current_d = final_cropped_img.shape[:3]
            
            if (current_h, current_w, current_d) != (target_h, target_w, target_d):
                # Center crop or pad to exact size
                padded_img = np.zeros((target_h, target_w, target_d, 4))
                padded_mask = np.zeros((target_h, target_w, target_d, 4))
                
                # Calculate copy region
                copy_h = min(current_h, target_h)
                copy_w = min(current_w, target_w)
                copy_d = min(current_d, target_d)
                
                # Center the data
                start_h = (target_h - copy_h) // 2
                start_w = (target_w - copy_w) // 2
                start_d = (target_d - copy_d) // 2
                
                padded_img[start_h:start_h+copy_h, start_w:start_w+copy_w, start_d:start_d+copy_d] = \
                    final_cropped_img[:copy_h, :copy_w, :copy_d]
                padded_mask[start_h:start_h+copy_h, start_w:start_w+copy_w, start_d:start_d+copy_d] = \
                    cropped_mask[:copy_h, :copy_w, :copy_d]
                
                final_cropped_img = padded_img
                cropped_mask = padded_mask
                
                # Update crop coordinates
                crop_coords['padding'] = {
                    'h_start': start_h, 'h_end': start_h + copy_h,
                    'w_start': start_w, 'w_end': start_w + copy_w,
                    'd_start': start_d, 'd_end': start_d + copy_d
                }
            
            logger.info(f"Final cropped image shape: {final_cropped_img.shape}")
            
            # Save intermediate results
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    final_cropped_img,
                    f"{patient_id}_cropped_training_style.nii.gz"
                )
            
            return final_cropped_img, cropped_mask, crop_coords
            
        except Exception as e:
            logger.error(f"Error in training-style cropping: {e}")
            raise

    def postprocess_segmentation_training_style(self, cropped_seg, crop_coords, original_shape, patient_id):
        """
        Postprocess segmentation from training-style processing back to original BraTS space.
        This handles both the ROI crop reconstruction and the 128x128x128 to 240x240x155 expansion.
        """
        logger.info(f"Postprocessing segmentation for {patient_id} from training-style processing")
        
        try:
            # Step 1: Reconstruct to 128x128x128 space (training space)
            training_shape = crop_coords['original_shape']  # Should be (128, 128, 128)
            logger.info(f"Reconstructing to training space: {training_shape}")
            
            # Initialize segmentation in training space
            training_segmentation = np.zeros(training_shape, dtype=cropped_seg.dtype)
            
            # Handle padding if it was applied during cropping
            if 'padding' in crop_coords:
                # Remove padding from cropped segmentation first
                padding = crop_coords['padding']
                unpadded_seg = cropped_seg[
                    padding['h_start']:padding['h_end'],
                    padding['w_start']:padding['w_end'], 
                    padding['d_start']:padding['d_end']
                ]
                logger.info(f"Removed padding, unpadded shape: {unpadded_seg.shape}")
            else:
                unpadded_seg = cropped_seg
            
            # Place back at ROI location in training space
            h_start = crop_coords['h_start']
            h_end = crop_coords['h_end']
            w_start = crop_coords['w_start'] 
            w_end = crop_coords['w_end']
            d_start = crop_coords['d_start']
            d_end = crop_coords['d_end']
            
            # Ensure we don't exceed training space boundaries
            actual_h_end = min(h_start + unpadded_seg.shape[0], training_shape[0])
            actual_w_end = min(w_start + unpadded_seg.shape[1], training_shape[1])
            actual_d_end = min(d_start + unpadded_seg.shape[2], training_shape[2])
            
            # Calculate how much to copy
            copy_h = actual_h_end - h_start
            copy_w = actual_w_end - w_start  
            copy_d = actual_d_end - d_start
            
            # Place segmentation in training space
            training_segmentation[h_start:actual_h_end, w_start:actual_w_end, d_start:actual_d_end] = \
                unpadded_seg[:copy_h, :copy_w, :copy_d]
            
            logger.info(f"Reconstructed segmentation to training space: {training_segmentation.shape}")
            
            # Step 2: Expand from training space (128x128x128) back to original BraTS space (240x240x155)
            final_segmentation = np.zeros(original_shape, dtype=cropped_seg.dtype)
            
            # Place training space segmentation back into original position [56:184, 56:184, 13:141]
            final_segmentation[56:184, 56:184, 13:141] = training_segmentation
            
            logger.info(f"Final segmentation shape: {final_segmentation.shape}")
            logger.info(f"Final segmentation unique values: {np.unique(final_segmentation)}")
            
            # Calculate tumor volume statistics
            non_bg_voxels = np.sum(final_segmentation > 0)
            total_voxels = final_segmentation.size
            tumor_percentage = (non_bg_voxels / total_voxels) * 100
            logger.info(f"Tumor voxels: {non_bg_voxels}/{total_voxels} ({tumor_percentage:.2f}%)")
            
            return final_segmentation
            
        except Exception as e:
            logger.error(f"Error in training-style postprocessing: {e}")
            raise

    def preprocess_data(self, data_dict, patient_id):
        """Preprocess MRI data following the training pipeline."""
        logger.info(f"Preprocessing data for {patient_id}")
        
        t1 = data_dict['t1']
        t1ce = data_dict['t1ce'] 
        t2 = data_dict['t2']
        flair = data_dict['flair']
        
        logger.info(f"Original image shapes: T1={t1.shape}, T1CE={t1ce.shape}, T2={t2.shape}, FLAIR={flair.shape}")
        
        try:
            # Use our own normalization that preserves the original shape
            normalized_data = self.normalize_mri_data_preserve_shape(t1, t1ce, t2, flair)
            logger.info(f"Normalized data shape: {normalized_data.shape}")
            
            # Create dummy mask for compatibility 
            dummy_mask = np.zeros((*normalized_data.shape[:3], 4))
            
            # Save intermediate normalized data if requested
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    normalized_data, 
                    f"{patient_id}_normalized.nii.gz",
                    reference_header=None
                )
            
            return normalized_data, dummy_mask
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def normalize_mri_data_preserve_shape(self, t1, t1ce, t2, flair):
        """Normalize MRI data using z-score but preserve original shape."""
        logger.info("Normalizing MRI data while preserving original shape")
        
        # Apply z-score normalization to each modality
        def calc_z_score_preserve(img):
            """Calculate z-score while preserving shape."""
            img_normalized = img.copy().astype(np.float32)
            
            # Only normalize non-zero voxels
            non_zero_mask = img_normalized != 0
            if np.sum(non_zero_mask) > 0:
                avg_pixel_value = np.mean(img_normalized[non_zero_mask])
                sd_pixel_value = np.std(img_normalized[non_zero_mask])
                
                if sd_pixel_value > 0:
                    img_normalized[non_zero_mask] = (img_normalized[non_zero_mask] - avg_pixel_value) / sd_pixel_value
            
            return img_normalized
        
        # Normalize each modality
        t1_norm = calc_z_score_preserve(t1)
        t1ce_norm = calc_z_score_preserve(t1ce)
        t2_norm = calc_z_score_preserve(t2)
        flair_norm = calc_z_score_preserve(flair)
        
        # Stack modalities along the last axis
        normalized_data = np.stack([flair_norm, t1ce_norm, t1_norm, t2_norm], axis=3)
        
        logger.info(f"Normalized data shape: {normalized_data.shape}")
        return normalized_data
    
    def predict_binary_mask(self, normalized_data, patient_id):
        """Generate binary mask using the binary model."""
        logger.info(f"Generating binary mask for {patient_id}")
        
        try:
            original_shape = normalized_data.shape[:3]
            logger.info(f"Original normalized data shape: {original_shape}")
            
            # The binary model expects 128x128x128 input
            # We need to crop the center region for prediction
            target_shape = (128, 128, 128)
            
            if original_shape == target_shape:
                # Already the right size
                cropped_for_binary = normalized_data
                crop_info = None
            else:
                # Crop center region to 128x128x128
                h, w, d = original_shape
                
                # Calculate crop boundaries (center crop)
                start_h = max(0, (h - 128) // 2)
                end_h = min(h, start_h + 128)
                start_w = max(0, (w - 128) // 2)
                end_w = min(w, start_w + 128)
                start_d = max(0, (d - 128) // 2)
                end_d = min(d, start_d + 128)
                
                crop_info = {
                    'h_start': start_h, 'h_end': end_h,
                    'w_start': start_w, 'w_end': end_w,
                    'd_start': start_d, 'd_end': end_d,
                    'original_shape': original_shape
                }
                
                cropped_for_binary = normalized_data[start_h:end_h, start_w:end_w, start_d:end_d]
                logger.info(f"Cropped for binary model: {cropped_for_binary.shape}")
                
                # Pad if needed to ensure exactly 128x128x128
                if cropped_for_binary.shape[:3] != target_shape:
                    padded = np.zeros((128, 128, 128, 4))
                    
                    actual_h, actual_w, actual_d = cropped_for_binary.shape[:3]
                    pad_h = (128 - actual_h) // 2
                    pad_w = (128 - actual_w) // 2
                    pad_d = (128 - actual_d) // 2
                    
                    padded[pad_h:pad_h+actual_h, pad_w:pad_w+actual_w, pad_d:pad_d+actual_d] = cropped_for_binary
                    cropped_for_binary = padded
                    
                    # Update crop info to account for padding
                    crop_info['padding'] = {
                        'h_start': pad_h, 'h_end': pad_h + actual_h,
                        'w_start': pad_w, 'w_end': pad_w + actual_w,
                        'd_start': pad_d, 'd_end': pad_d + actual_d
                    }
            
            # Prepare input for binary model
            img_input = np.expand_dims(cropped_for_binary, axis=0)
            
            # Predict on the 128x128x128 crop
            binary_pred = self.binary_model.predict(img_input, verbose=0)
            binary_mask_crop = binary_pred[0, :, :, :, 0]
            
            # Threshold to get binary mask
            binary_mask_crop_thresh = (binary_mask_crop > 0.5).astype(np.uint8)
            
            logger.info(f"Binary mask crop shape: {binary_mask_crop_thresh.shape}")
            logger.info(f"Binary mask crop unique values: {np.unique(binary_mask_crop_thresh)}")
            
            # Reconstruct full binary mask if we cropped
            if crop_info is not None:
                binary_mask_full = np.zeros(original_shape, dtype=np.uint8)
                
                # Handle padding if it was applied
                if 'padding' in crop_info:
                    padding = crop_info['padding']
                    unpadded_mask = binary_mask_crop_thresh[
                        padding['h_start']:padding['h_end'],
                        padding['w_start']:padding['w_end'],
                        padding['d_start']:padding['d_end']
                    ]
                else:
                    unpadded_mask = binary_mask_crop_thresh
                
                # Place back in original position
                h_start = crop_info['h_start']
                w_start = crop_info['w_start']
                d_start = crop_info['d_start']
                
                h_end = min(h_start + unpadded_mask.shape[0], original_shape[0])
                w_end = min(w_start + unpadded_mask.shape[1], original_shape[1])
                d_end = min(d_start + unpadded_mask.shape[2], original_shape[2])
                
                copy_h = h_end - h_start
                copy_w = w_end - w_start
                copy_d = d_end - d_start
                
                binary_mask_full[h_start:h_end, w_start:w_end, d_start:d_end] = \
                    unpadded_mask[:copy_h, :copy_w, :copy_d]
                
                binary_mask = binary_mask_full
                logger.info(f"Reconstructed binary mask to original shape: {binary_mask.shape}")
            else:
                binary_mask = binary_mask_crop_thresh
            
            logger.info(f"Final binary mask shape: {binary_mask.shape}")
            logger.info(f"Final binary mask unique values: {np.unique(binary_mask)}")
            
            # Save intermediate binary mask
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    binary_mask,
                    f"{patient_id}_binary_mask.nii.gz"
                )
                
                # Also save raw probabilities (reconstructed)
                if crop_info is not None:
                    binary_prob_full = np.zeros(original_shape, dtype=np.float32)
                    
                    if 'padding' in crop_info:
                        padding = crop_info['padding']
                        unpadded_prob = binary_mask_crop[
                            padding['h_start']:padding['h_end'],
                            padding['w_start']:padding['w_end'],
                            padding['d_start']:padding['d_end']
                        ]
                    else:
                        unpadded_prob = binary_mask_crop
                    
                    h_start = crop_info['h_start']
                    w_start = crop_info['w_start']
                    d_start = crop_info['d_start']
                    
                    h_end = min(h_start + unpadded_prob.shape[0], original_shape[0])
                    w_end = min(w_start + unpadded_prob.shape[1], original_shape[1])
                    d_end = min(d_start + unpadded_prob.shape[2], original_shape[2])
                    
                    copy_h = h_end - h_start
                    copy_w = w_end - w_start
                    copy_d = d_end - d_start
                    
                    binary_prob_full[h_start:h_end, w_start:w_end, d_start:d_end] = \
                        unpadded_prob[:copy_h, :copy_w, :copy_d]
                    
                    self.save_intermediate_nifti(
                        binary_prob_full,
                        f"{patient_id}_binary_prob.nii.gz"
                    )
                else:
                    self.save_intermediate_nifti(
                        binary_mask_crop,
                        f"{patient_id}_binary_prob.nii.gz"
                    )
            
            return binary_mask
            
        except Exception as e:
            logger.error(f"Error in binary prediction: {e}")
            raise
    
    def crop_with_binary_mask(self, normalized_data, mask, binary_mask, patient_id):
        """Crop the data using ROI detection from binary mask."""
        logger.info(f"Cropping data using binary mask for {patient_id}")
        
        try:
            # Use a safer approach instead of roi_crop function
            # Since the binary mask might not have exact 1s, use thresholding
            binary_thresh = (binary_mask > 0.5).astype(np.uint8)
            
            # Find coordinates where tumor is present
            loc = np.where(binary_thresh == 1)
            
            # Store original cropping coordinates for reconstruction
            crop_coords = {}
            
            if len(loc[0]) == 0:
                # No tumor detected, use center crop as fallback
                logger.warning(f"No tumor region detected for {patient_id}, using center crop")
                h, w, d = normalized_data.shape[:3]
                
                # Center crop to 48x48x128
                start_h = max(0, (h - 48) // 2)
                end_h = min(h, start_h + 48)
                start_w = max(0, (w - 48) // 2) 
                end_w = min(w, start_w + 48)
                start_d = 0
                end_d = min(d, 128)
                
                crop_coords = {
                    'h_start': start_h, 'h_end': end_h,
                    'w_start': start_w, 'w_end': end_w,
                    'd_start': start_d, 'd_end': end_d,
                    'original_shape': normalized_data.shape[:3]
                }
                
                cropped_img = normalized_data[start_h:end_h, start_w:end_w, start_d:end_d]
                cropped_mask = mask[start_h:end_h, start_w:end_w, start_d:end_d]
                binary_cropped = binary_thresh[start_h:end_h, start_w:end_w, start_d:end_d]
                
            else:
                # Tumor detected, crop around it
                thresh = 12
                a = max(0, np.amin(loc[0]) - thresh)
                b = min(128, np.amax(loc[0]) + thresh)
                c = max(0, np.amin(loc[1]) - thresh)
                d = min(128, np.amax(loc[1]) + thresh)

                # Ensure minimum size of 48x48 and maximum of 48x48
                crop_h = min(48, b - a)
                crop_w = min(48, d - c)
                
                # Adjust bounds to get exactly 48x48
                center_h = (a + b) // 2
                center_w = (c + d) // 2
                
                a = max(0, center_h - 24)
                b = min(128, a + 48)
                c = max(0, center_w - 24) 
                d = min(128, c + 48)
                
                # Final adjustment if we hit boundaries
                if b - a < 48:
                    if a == 0:
                        b = 48
                    else:
                        a = 128 - 48
                        b = 128
                        
                if d - c < 48:
                    if c == 0:
                        d = 48
                    else:
                        c = 128 - 48
                        d = 128

                # Store cropping coordinates
                crop_coords = {
                    'h_start': a, 'h_end': b,
                    'w_start': c, 'w_end': d,
                    'd_start': 0, 'd_end': 128,
                    'original_shape': normalized_data.shape[:3]
                }

                # Crop the data to 48x48x128
                cropped_img = normalized_data[a:b, c:d, :128]  # Ensure depth is 128
                cropped_mask = mask[a:b, c:d, :128]
                binary_cropped = binary_thresh[a:b, c:d, :128]
            
            # Ensure exactly 48x48x128 size
            target_shape = (48, 48, 128)
            if cropped_img.shape[:3] != target_shape:
                # Pad or crop to exact size
                padded_img = np.zeros((48, 48, 128, 4))
                padded_mask = np.zeros((48, 48, 128, 4))
                padded_binary = np.zeros((48, 48, 128))
                
                # Calculate how much to copy
                copy_h = min(cropped_img.shape[0], 48)
                copy_w = min(cropped_img.shape[1], 48)
                copy_d = min(cropped_img.shape[2], 128)
                
                # Copy to center if smaller, or crop if larger
                start_h = (48 - copy_h) // 2 if copy_h < 48 else 0
                start_w = (48 - copy_w) // 2 if copy_w < 48 else 0
                start_d = (128 - copy_d) // 2 if copy_d < 128 else 0
                
                padded_img[start_h:start_h+copy_h, start_w:start_w+copy_w, start_d:start_d+copy_d] = \
                    cropped_img[:copy_h, :copy_w, :copy_d]
                padded_mask[start_h:start_h+copy_h, start_w:start_w+copy_w, start_d:start_d+copy_d] = \
                    cropped_mask[:copy_h, :copy_w, :copy_d]
                padded_binary[start_h:start_h+copy_h, start_w:start_w+copy_w, start_d:start_d+copy_d] = \
                    binary_cropped[:copy_h, :copy_w, :copy_d]
                
                cropped_img = padded_img
                cropped_mask = padded_mask
                binary_cropped = padded_binary
                
                # Update crop coordinates to account for padding
                crop_coords['padding'] = {
                    'h_start': start_h, 'h_end': start_h + copy_h,
                    'w_start': start_w, 'w_end': start_w + copy_w,
                    'd_start': start_d, 'd_end': start_d + copy_d
                }
            
            # Add binary mask as 5th channel for cropped data (48x48x128x5)
            binary_expanded = np.expand_dims(binary_cropped, axis=-1)
            cropped_img_with_binary = np.concatenate([cropped_img, binary_expanded], axis=-1)
            
            logger.info(f"Cropped image shape: {cropped_img_with_binary.shape}")
            logger.info(f"Cropped mask shape: {cropped_mask.shape}")
            logger.info(f"Crop coordinates: {crop_coords}")
            
            # Save intermediate cropped data
            if self.save_intermediate:
                # Save only the first 4 channels (the MRI modalities)
                self.save_intermediate_nifti(
                    cropped_img,  # This is now 48x48x128x4
                    f"{patient_id}_cropped.nii.gz"
                )
            
            return cropped_img_with_binary, cropped_mask, crop_coords
            
        except Exception as e:
            logger.error(f"Error in cropping: {e}")
            raise
    
    def predict_multiclass(self, cropped_data, patient_id):
        """Generate multiclass segmentation using the attention model."""
        logger.info(f"Generating multiclass segmentation for {patient_id}")
        
        try:
            logger.info(f"Input cropped data shape: {cropped_data.shape}")
            
            # The multiclass model expects 4 channels (MRI modalities only)
            if cropped_data.shape[-1] == 4:
                # Training-style preprocessing: already 4 channels
                mri_data = cropped_data
                logger.info(f"Using 4-channel MRI data shape: {mri_data.shape}")
            elif cropped_data.shape[-1] == 5:
                # Original preprocessing: extract only the first 4 channels (exclude binary mask)
                mri_data = cropped_data[:, :, :, :4]
                logger.info(f"Extracted 4-channel MRI data from 5-channel input: {mri_data.shape}")
            else:
                logger.error(f"Unexpected cropped data shape: {cropped_data.shape}")
                raise ValueError(f"Expected 4 or 5 channels, got {cropped_data.shape[-1]} channels")
            
            # Verify shape is exactly what multiclass model expects
            if mri_data.shape != (48, 48, 128, 4):
                logger.error(f"MRI data shape {mri_data.shape} doesn't match expected (48, 48, 128, 4)")
                raise ValueError(f"MRI data must be exactly (48, 48, 128, 4), got {mri_data.shape}")
            
            # Prepare input for multiclass model (should be 48x48x128x4)
            img_input = np.expand_dims(mri_data, axis=0)
            logger.info(f"Model input shape: {img_input.shape}")
            
            # Predict
            multiclass_pred = self.multiclass_model.predict(img_input, verbose=0)
            multiclass_prob = multiclass_pred[0]  # Shape: (48, 48, 128, 4)
            
            logger.info(f"Multiclass prediction shape: {multiclass_pred.shape}")
            logger.info(f"Multiclass probabilities shape: {multiclass_prob.shape}")
            
            # Convert probabilities to class labels
            multiclass_seg = np.argmax(multiclass_prob, axis=-1)
            
            logger.info(f"Multiclass segmentation shape: {multiclass_seg.shape}")
            logger.info(f"Multiclass unique values: {np.unique(multiclass_seg)}")
            
            # Save intermediate multiclass results
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    multiclass_seg,
                    f"{patient_id}_multiclass_raw.nii.gz"
                )
                
                # Save probabilities for each class
                for i in range(multiclass_prob.shape[-1]):
                    self.save_intermediate_nifti(
                        multiclass_prob[:, :, :, i],
                        f"{patient_id}_prob_class_{i}.nii.gz"
                    )
            
            return multiclass_seg, multiclass_prob
            
        except Exception as e:
            logger.error(f"Error in multiclass prediction: {e}")
            raise
    
    def convert_to_brats_labels(self, segmentation):
        """Convert model output to BraTS label format."""
        # Model outputs 0, 1, 2, 3 -> Convert to BraTS format 0, 1, 2, 4
        brats_seg = segmentation.copy()
        brats_seg[segmentation == 3] = 4  # ET class becomes label 4
        return brats_seg
    
    def postprocess_segmentation(self, cropped_seg, crop_coords, patient_id):
        """Postprocess segmentation to original size using stored crop coordinates."""
        logger.info(f"Postprocessing segmentation for {patient_id}")
        
        try:
            original_shape = crop_coords['original_shape']
            logger.info(f"Reconstructing to original shape: {original_shape}")
            
            # Initialize full-size segmentation with background (0)
            full_segmentation = np.zeros(original_shape, dtype=cropped_seg.dtype)
            
            # Handle padding if it was applied during cropping
            if 'padding' in crop_coords:
                # Remove padding from cropped segmentation first
                padding = crop_coords['padding']
                unpadded_seg = cropped_seg[
                    padding['h_start']:padding['h_end'],
                    padding['w_start']:padding['w_end'], 
                    padding['d_start']:padding['d_end']
                ]
                logger.info(f"Removed padding, unpadded shape: {unpadded_seg.shape}")
            else:
                unpadded_seg = cropped_seg
            
            # Place the segmentation back at the original crop location
            h_start = crop_coords['h_start']
            h_end = crop_coords['h_end']
            w_start = crop_coords['w_start'] 
            w_end = crop_coords['w_end']
            d_start = crop_coords['d_start']
            d_end = crop_coords['d_end']
            
            # Ensure we don't exceed original boundaries
            actual_h_end = min(h_start + unpadded_seg.shape[0], original_shape[0])
            actual_w_end = min(w_start + unpadded_seg.shape[1], original_shape[1])
            actual_d_end = min(d_start + unpadded_seg.shape[2], original_shape[2])
            
            # Calculate how much of the unpadded segmentation to copy
            copy_h = actual_h_end - h_start
            copy_w = actual_w_end - w_start  
            copy_d = actual_d_end - d_start
            
            logger.info(f"Placing segmentation at [{h_start}:{actual_h_end}, {w_start}:{actual_w_end}, {d_start}:{actual_d_end}]")
            logger.info(f"Copying from unpadded_seg: [{0}:{copy_h}, {0}:{copy_w}, {0}:{copy_d}]")
            
            # Place the segmentation
            full_segmentation[h_start:actual_h_end, w_start:actual_w_end, d_start:actual_d_end] = \
                unpadded_seg[:copy_h, :copy_w, :copy_d]
            
            logger.info(f"Final segmentation shape: {full_segmentation.shape}")
            logger.info(f"Final segmentation unique values: {np.unique(full_segmentation)}")
            
            # Calculate tumor volume statistics
            non_bg_voxels = np.sum(full_segmentation > 0)
            total_voxels = full_segmentation.size
            tumor_percentage = (non_bg_voxels / total_voxels) * 100
            logger.info(f"Tumor voxels: {non_bg_voxels}/{total_voxels} ({tumor_percentage:.2f}%)")
            
            return full_segmentation
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            raise
    
    def save_intermediate_nifti(self, data, filename, reference_header=None):
        """Save intermediate results as NIfTI files."""
        try:
            output_path = self.output_dir / 'intermediate' / filename
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(data.astype(np.int8), affine=np.eye(4), header=reference_header)
            nib.save(nii_img, output_path)
            
            logger.debug(f"Saved intermediate result: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate file {filename}: {e}")
    
    def save_final_prediction(self, segmentation, patient_id, reference_header=None):
        """Save final prediction as NIfTI file."""
        try:
            output_path = self.output_dir / 'predictions' / f"{patient_id}_prediction.nii.gz"
            
            # Create NIfTI image with proper header
            nii_img = nib.Nifti1Image(segmentation.astype(np.uint8), affine=np.eye(4), header=reference_header)
            nib.save(nii_img, output_path)
            
            logger.info(f"Saved final prediction: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving final prediction: {e}")
            raise

    def save_resampled_predictions(self, binary_mask, multiclass_seg, ants_images, patient_id):
        """Save resampled predictions using ANTs for proper spatial alignment."""
        logger.info(f"Saving resampled predictions for {patient_id}")
        
        resampled_paths = {}
        
        try:
            # Use ground truth segmentation as reference if available, otherwise use T1
            if 'seg' in ants_images:
                reference_img = ants_images['seg']
                reference_name = "ground_truth"
            else:
                reference_img = ants_images['t1']
                reference_name = "T1"
            
            logger.info(f"Using {reference_name} as reference for resampling")
            logger.info(f"Reference image shape: {reference_img.shape}")
            logger.info(f"Reference image spacing: {reference_img.spacing}")
            
            # Resample binary mask
            binary_resampled, binary_ants = self.resample_to_reference(
                binary_mask, reference_img, "binary_mask"
            )
            
            # Save resampled binary mask
            binary_path = self.output_dir / 'predictions_resampled' / f"{patient_id}_binary_resampled.nii.gz"
            self.save_ants_image(binary_ants, binary_path)
            resampled_paths['binary'] = str(binary_path)
            
            # Resample multiclass segmentation
            multiclass_resampled, multiclass_ants = self.resample_to_reference(
                multiclass_seg, reference_img, "multiclass_segmentation"
            )
            
            # Save resampled multiclass segmentation
            multiclass_path = self.output_dir / 'predictions_resampled' / f"{patient_id}_multiclass_resampled.nii.gz"
            self.save_ants_image(multiclass_ants, multiclass_path)
            resampled_paths['multiclass'] = str(multiclass_path)
            
            # Save intermediate resampled results if requested
            if self.save_intermediate:
                binary_intermediate = self.output_dir / 'intermediate' / f"{patient_id}_binary_resampled.nii.gz"
                multiclass_intermediate = self.output_dir / 'intermediate' / f"{patient_id}_multiclass_resampled.nii.gz"
                
                self.save_ants_image(binary_ants, binary_intermediate)
                self.save_ants_image(multiclass_ants, multiclass_intermediate)
            
            logger.info(f"Successfully saved resampled predictions:")
            logger.info(f"  Binary: {binary_path}")
            logger.info(f"  Multiclass: {multiclass_path}")
            
            return resampled_paths, {
                'binary_resampled': binary_resampled,
                'multiclass_resampled': multiclass_resampled,
                'reference_used': reference_name
            }
            
        except Exception as e:
            logger.error(f"Error saving resampled predictions: {e}")
            raise
    
    def generate_summary_report(self, patient_id, segmentation, processing_time, resampled_data=None, evaluation_results=None):
        """Generate a summary report for the processed patient."""
        try:
            # Calculate statistics for original segmentation
            unique_labels, counts = np.unique(segmentation, return_counts=True)
            total_voxels = segmentation.size
            
            report = {
                'patient_id': patient_id,
                'processing_time_seconds': processing_time,
                'original_segmentation': {
                    'shape': segmentation.shape,
                    'total_voxels': int(total_voxels),
                    'label_statistics': {}
                }
            }
            
            for label, count in zip(unique_labels, counts):
                label_name = self.label_mapping.get(int(label), f"Unknown_{label}")
                percentage = (count / total_voxels) * 100
                report['original_segmentation']['label_statistics'][int(label)] = {
                    'name': label_name,
                    'voxel_count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            # Add resampled data statistics if available
            if resampled_data:
                report['resampling'] = {
                    'reference_used': resampled_data['reference_used']
                }
                
                # Statistics for resampled binary mask
                if 'binary_resampled' in resampled_data:
                    binary_unique, binary_counts = np.unique(resampled_data['binary_resampled'], return_counts=True)
                    binary_total = resampled_data['binary_resampled'].size
                    
                    report['resampling']['binary_mask'] = {
                        'shape': resampled_data['binary_resampled'].shape,
                        'total_voxels': int(binary_total),
                        'tumor_voxels': int(np.sum(resampled_data['binary_resampled'] > 0)),
                        'tumor_percentage': round((np.sum(resampled_data['binary_resampled'] > 0) / binary_total) * 100, 2)
                    }
                
                # Statistics for resampled multiclass segmentation
                if 'multiclass_resampled' in resampled_data:
                    mc_unique, mc_counts = np.unique(resampled_data['multiclass_resampled'], return_counts=True)
                    mc_total = resampled_data['multiclass_resampled'].size
                    
                    report['resampling']['multiclass_segmentation'] = {
                        'shape': resampled_data['multiclass_resampled'].shape,
                        'total_voxels': int(mc_total),
                        'label_statistics': {}
                    }
                    
                    for label, count in zip(mc_unique, mc_counts):
                        label_name = self.label_mapping.get(int(label), f"Unknown_{label}")
                        percentage = (count / mc_total) * 100
                        report['resampling']['multiclass_segmentation']['label_statistics'][int(label)] = {
                            'name': label_name,
                            'voxel_count': int(count),
                            'percentage': round(percentage, 2)
                        }
            
            # Add evaluation results if available
            if evaluation_results:
                report['evaluation'] = evaluation_results
            
            # Save report
            report_path = self.output_dir / 'logs' / f"{patient_id}_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Generated report: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def process_patient(self, patient_dir):
        """Process a single patient through the complete pipeline."""
        start_time = datetime.now()
        
        try:
            # Load patient data including ground truth
            data_dict, headers, ants_images, patient_id = self.load_patient_data(patient_dir)
            
            # Check if ground truth is available
            has_ground_truth = 'seg' in data_dict
            ground_truth = data_dict.get('seg', None)
            
            # Choose preprocessing pipeline
            if self.use_training_style_preprocessing:
                logger.info(f"Using TRAINING-STYLE preprocessing pipeline for {patient_id}")
                
                # Preprocess using training-style pipeline
                normalized_data, processed_mask, crop_indices = self.preprocess_data_training_style(data_dict, patient_id)
                
                # Generate binary mask (training-style: expects 128x128x128x4)
                binary_mask = self.predict_binary_mask_training_style(normalized_data, patient_id)
                
                # Crop using binary mask (training-style: ROI crop to 48x48x128x4)
                cropped_data, cropped_mask, crop_coords = self.crop_with_binary_mask_training_style(
                    normalized_data, processed_mask, binary_mask, patient_id
                )
                
                # Generate multiclass segmentation
                multiclass_seg, multiclass_prob = self.predict_multiclass(cropped_data, patient_id)
                
                # Convert to BraTS format (0, 1, 2, 4) - convert label 3 back to 4
                multiclass_seg[multiclass_seg == 3] = 4
                
                # Postprocess to original BraTS size (training-style: 48x48x128 -> 128x128x128 -> 240x240x155)
                original_shape = data_dict['t1'].shape  # Original BraTS shape
                final_segmentation = self.postprocess_segmentation_training_style(
                    multiclass_seg, crop_coords, original_shape, patient_id
                )
                
                # Also reconstruct binary mask to original space for evaluation
                binary_mask_full = np.zeros(original_shape, dtype=np.uint8)
                binary_mask_full[56:184, 56:184, 13:141] = binary_mask
                
            else:
                logger.info(f"Using ORIGINAL preprocessing pipeline for {patient_id}")
                
                # Original preprocessing pipeline
                normalized_data, processed_mask = self.preprocess_data(data_dict, patient_id)
                
                # Generate binary mask
                binary_mask_full = self.predict_binary_mask(normalized_data, patient_id)
                
                # Crop using binary mask
                cropped_data, cropped_mask, crop_coords = self.crop_with_binary_mask(
                    normalized_data, processed_mask, binary_mask_full, patient_id
                )
                
                # Generate multiclass segmentation
                multiclass_seg, multiclass_prob = self.predict_multiclass(cropped_data, patient_id)
                
                # Convert to BraTS format (0, 1, 2, 4)
                multiclass_brats = self.convert_to_brats_labels(multiclass_seg)
                
                # Postprocess to original size
                final_segmentation = self.postprocess_segmentation(
                    multiclass_brats, crop_coords, patient_id
                )
            
            # Evaluate predictions against ground truth if available
            evaluation_results = None
            if has_ground_truth:
                try:
                    evaluation_results = self.evaluate_predictions(
                        binary_mask_full, final_segmentation, ground_truth, patient_id
                    )
                    logger.info(f"Evaluation completed for {patient_id}")
                except Exception as e:
                    logger.warning(f"Evaluation failed for {patient_id}: {e}")
                    evaluation_results = {
                        'has_ground_truth': True,
                        'evaluation_error': str(e)
                    }
            else:
                logger.info(f"No ground truth available for {patient_id}, skipping evaluation")
                evaluation_results = {'has_ground_truth': False}
            
            # Save final prediction in original processing space
            prediction_path = self.save_final_prediction(
                final_segmentation, patient_id, headers['t1']
            )
            
            # Resample predictions to ground truth space using ANTs
            resampled_paths = {}
            resampled_data = None
            
            if self.use_ground_truth_space:
                try:
                    resampled_paths, resampled_data = self.save_resampled_predictions(
                        binary_mask_full, final_segmentation, ants_images, patient_id
                    )
                    logger.info("Successfully resampled predictions to reference space")
                    
                    # Evaluate resampled predictions if ground truth is available
                    if has_ground_truth and 'evaluation_error' not in evaluation_results:
                        try:
                            # Get ground truth in the same space as resampled predictions
                            if 'seg' in ants_images:
                                gt_resampled = ants_images['seg'].numpy()
                            else:
                                gt_resampled = ground_truth
                            
                            resampled_evaluation = self.evaluate_predictions(
                                resampled_data['binary_resampled'],
                                resampled_data['multiclass_resampled'],
                                gt_resampled,
                                f"{patient_id}_resampled"
                            )
                            
                            # Add resampled evaluation to results
                            evaluation_results['resampled_evaluation'] = {
                                'binary_evaluation': resampled_evaluation['binary_evaluation'],
                                'multiclass_evaluation': resampled_evaluation['multiclass_evaluation']
                            }
                            
                        except Exception as e:
                            logger.warning(f"Resampled evaluation failed for {patient_id}: {e}")
                            evaluation_results['resampled_evaluation_error'] = str(e)
                            
                except Exception as e:
                    logger.warning(f"Failed to resample predictions: {e}")
                    logger.info("Continuing with original predictions only")
            
            # Generate report with evaluation information
            processing_time = (datetime.now() - start_time).total_seconds()
            report = self.generate_summary_report(
                patient_id, final_segmentation, processing_time, resampled_data, evaluation_results
            )
            
            logger.info(f"Successfully processed {patient_id} in {processing_time:.2f} seconds")
            
            result = {
                'patient_id': patient_id,
                'success': True,
                'prediction_path': str(prediction_path),
                'processing_time': processing_time,
                'report': report,
                'evaluation': evaluation_results,
                'preprocessing_style': 'training-style' if self.use_training_style_preprocessing else 'original'
            }
            
            # Add resampled paths if available
            if resampled_paths:
                result['resampled_paths'] = resampled_paths
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to process {patient_dir}: {e}")
            
            return {
                'patient_id': getattr(self, 'current_patient_id', str(patient_dir)),
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def run_inference(self):
        """Run inference on all patients in test_dir."""
        logger.info("Starting inference on test data...")
        
        # Load models
        self.load_models()
        
        # Find all patient directories
        patient_dirs = [d for d in self.test_dir.iterdir() if d.is_dir()]
        
        if not patient_dirs:
            logger.warning(f"No patient directories found in {self.test_dir}")
            return
        
        logger.info(f"Found {len(patient_dirs)} patient directories")
        
        results = []
        successful = 0
        failed = 0
        
        for patient_dir in patient_dirs:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {patient_dir.name}")
            logger.info(f"{'='*50}")
            
            result = self.process_patient(patient_dir)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Generate overall summary
        summary = {
            'total_patients': len(patient_dirs),
            'successful': successful,
            'failed': failed,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_dir / 'inference_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"INFERENCE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total patients: {len(patient_dirs)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary: {summary_path}")


def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Inference with ANTs Resampling')
    parser.add_argument('--test_dir', default='test_dir', 
                       help='Directory containing test NIfTI files')
    parser.add_argument('--output_dir', default='test_results',
                       help='Directory to save results')
    parser.add_argument('--binary_weights', default='/data/aniket/BrainTumorSegmentation/weights/BinaryWeights.hdf5',
                       help='Path to binary model weights')
    parser.add_argument('--multiclass_weights', default='/data/aniket/BrainTumorSegmentation/weights/weights_1000instances.h5',
                       help='Path to multiclass model weights')
    parser.add_argument('--no_intermediate', action='store_true',
                       help='Skip saving intermediate results')
    parser.add_argument('--no_resampling', action='store_true',
                       help='Skip ANTs resampling to ground truth space')
    parser.add_argument('--use_original_preprocessing', action='store_true',
                       help='Use original preprocessing instead of training-style preprocessing')
    
    args = parser.parse_args()
    
    # Create tester
    tester = BrainTumorSegmentationTester(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        binary_weights=args.binary_weights,
        multiclass_weights=args.multiclass_weights,
        save_intermediate=not args.no_intermediate,
        use_ground_truth_space=not args.no_resampling,
        use_training_style_preprocessing=not args.use_original_preprocessing
    )
    
    # Run inference
    try:
        tester.run_inference()
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main() 