#!/usr/bin/env python3
"""
Simplified Brain Tumor Segmentation Test Script
===============================================

This script processes NIfTI images from test_dir and performs brain tumor segmentation
using the trained multiclass model directly, without the binary cropping step.
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

# Add utils to path
sys.path.append('utils')

from utils.models import attention_multiclass_model
from utils.preprocessing import normalize_mri_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_inference_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleBrainTumorSegmentationTester:
    """Simplified tester that works directly with the multiclass model."""
    
    def __init__(self, 
                 test_dir='test_dir',
                 output_dir='test_results_simple',
                 model_weights='weights/weights_1000instances.h5',
                 save_intermediate=True):
        
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.model_weights = model_weights
        self.save_intermediate = save_intermediate
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'intermediate').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize model
        self.model = None
        
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
    
    def load_model(self):
        """Load the trained multiclass model."""
        logger.info("Loading multiclass model...")
        
        try:
            self.model = attention_multiclass_model(48, 48, 128, 4, 4)
            
            if os.path.exists(self.model_weights):
                self.model.load_weights(self.model_weights)
                logger.info(f"Loaded model weights from {self.model_weights}")
            else:
                logger.warning(f"Model weights not found at {self.model_weights}. Using random weights.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_patient_data(self, patient_dir):
        """Load all MRI modalities for a patient."""
        patient_path = Path(patient_dir)
        patient_id = patient_path.name
        
        logger.info(f"Loading data for patient: {patient_id}")
        
        # Expected file patterns for BraTS data
        modalities = {
            't1': f"{patient_id}_t1.nii",
            't1ce': f"{patient_id}_t1ce.nii", 
            't2': f"{patient_id}_t2.nii",
            'flair': f"{patient_id}_flair.nii"
        }
        
        loaded_data = {}
        headers = {}
        
        for modality, filename in modalities.items():
            filepath = patient_path / filename
            
            if not filepath.exists():
                logger.error(f"Missing file: {filepath}")
                raise FileNotFoundError(f"Required file not found: {filepath}")
            
            try:
                nii_img = nib.load(filepath)
                loaded_data[modality] = nii_img.get_fdata()
                headers[modality] = nii_img.header
                logger.info(f"Loaded {modality}: {loaded_data[modality].shape}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                raise
        
        return loaded_data, headers, patient_id
    
    def preprocess_data(self, data_dict, patient_id):
        """Preprocess MRI data following the training pipeline."""
        logger.info(f"Preprocessing data for {patient_id}")
        
        t1 = data_dict['t1']
        t1ce = data_dict['t1ce'] 
        t2 = data_dict['t2']
        flair = data_dict['flair']
        
        # Create dummy mask for preprocessing (zeros)
        dummy_mask = np.zeros_like(t1)
        
        # Apply the normalization pipeline
        try:
            normalized_data, processed_mask = normalize_mri_data(
                t1, t1ce, t2, flair, dummy_mask
            )
            logger.info(f"Normalized data shape: {normalized_data.shape}")
            
            # Save intermediate normalized data if requested
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    normalized_data, 
                    f"{patient_id}_normalized.nii.gz"
                )
            
            return normalized_data, processed_mask
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def create_patches(self, data, patch_size=(48, 48, 128), stride=24):
        """Create overlapping patches from the full volume."""
        logger.info(f"Creating patches with size {patch_size} and stride {stride}")
        
        patches = []
        coordinates = []
        
        h, w, d, c = data.shape
        ph, pw, pd = patch_size
        
        # Generate patch coordinates
        for i in range(0, h - ph + 1, stride):
            for j in range(0, w - pw + 1, stride):
                for k in range(0, d - pd + 1, stride):
                    # Extract patch
                    patch = data[i:i+ph, j:j+pw, k:k+pd, :]
                    patches.append(patch)
                    coordinates.append((i, j, k))
        
        logger.info(f"Created {len(patches)} patches")
        return np.array(patches), coordinates
    
    def reconstruct_from_patches(self, patch_predictions, coordinates, original_shape, patch_size=(48, 48, 128)):
        """Reconstruct full volume from patch predictions."""
        logger.info("Reconstructing volume from patches")
        
        h, w, d = original_shape[:3]
        num_classes = patch_predictions.shape[-1]
        
        # Initialize accumulation arrays
        prediction_sum = np.zeros((h, w, d, num_classes))
        weight_sum = np.zeros((h, w, d))
        
        ph, pw, pd = patch_size
        
        for patch_pred, (i, j, k) in zip(patch_predictions, coordinates):
            # Add prediction to accumulation
            prediction_sum[i:i+ph, j:j+pw, k:k+pd, :] += patch_pred
            weight_sum[i:i+ph, j:j+pw, k:k+pd] += 1
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1)
        
        # Average the predictions
        final_prediction = prediction_sum / weight_sum[..., np.newaxis]
        
        return final_prediction
    
    def predict_segmentation(self, normalized_data, patient_id):
        """Generate segmentation using patch-based approach."""
        logger.info(f"Generating segmentation for {patient_id}")
        
        try:
            # Create patches
            patches, coordinates = self.create_patches(normalized_data)
            
            # Predict on all patches
            patch_predictions = []
            batch_size = 4  # Process patches in small batches to manage memory
            
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_pred = self.model.predict(batch, verbose=0)
                patch_predictions.extend(batch_pred)
            
            patch_predictions = np.array(patch_predictions)
            
            # Reconstruct full volume
            full_prediction = self.reconstruct_from_patches(
                patch_predictions, coordinates, normalized_data.shape
            )
            
            # Convert probabilities to class labels
            segmentation = np.argmax(full_prediction, axis=-1)
            
            logger.info(f"Segmentation shape: {segmentation.shape}")
            logger.info(f"Unique values: {np.unique(segmentation)}")
            
            # Convert to BraTS format (0, 1, 2, 4)
            brats_segmentation = self.convert_to_brats_labels(segmentation)
            
            # Save intermediate results
            if self.save_intermediate:
                self.save_intermediate_nifti(
                    segmentation,
                    f"{patient_id}_segmentation_raw.nii.gz"
                )
                
                self.save_intermediate_nifti(
                    brats_segmentation,
                    f"{patient_id}_segmentation_brats.nii.gz"
                )
                
                # Save probabilities for each class
                for i in range(full_prediction.shape[-1]):
                    self.save_intermediate_nifti(
                        full_prediction[:, :, :, i],
                        f"{patient_id}_prob_class_{i}.nii.gz"
                    )
            
            return brats_segmentation, full_prediction
            
        except Exception as e:
            logger.error(f"Error in segmentation prediction: {e}")
            raise
    
    def convert_to_brats_labels(self, segmentation):
        """Convert model output to BraTS label format."""
        # Model outputs 0, 1, 2, 3 -> Convert to BraTS format 0, 1, 2, 4
        brats_seg = segmentation.copy()
        brats_seg[segmentation == 3] = 4  # ET class becomes label 4
        return brats_seg
    
    def save_intermediate_nifti(self, data, filename, reference_header=None):
        """Save intermediate results as NIfTI files."""
        try:
            output_path = self.output_dir / 'intermediate' / filename
            
            # Choose appropriate data type based on content
            if 'prob_' in filename or 'normalized' in filename:
                # Probabilities and normalized data should be float32
                data_type = np.float32
            else:
                # Segmentation masks should be uint8
                data_type = np.uint8
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(data.astype(data_type), affine=np.eye(4), header=reference_header)
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
    
    def generate_summary_report(self, patient_id, segmentation, processing_time):
        """Generate a summary report for the processed patient."""
        try:
            # Calculate statistics
            unique_labels, counts = np.unique(segmentation, return_counts=True)
            total_voxels = segmentation.size
            
            report = {
                'patient_id': patient_id,
                'processing_time_seconds': processing_time,
                'segmentation_shape': segmentation.shape,
                'total_voxels': int(total_voxels),
                'label_statistics': {}
            }
            
            for label, count in zip(unique_labels, counts):
                label_name = self.label_mapping.get(int(label), f"Unknown_{label}")
                percentage = (count / total_voxels) * 100
                report['label_statistics'][int(label)] = {
                    'name': label_name,
                    'voxel_count': int(count),
                    'percentage': round(percentage, 2)
                }
            
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
            # Load patient data
            data_dict, headers, patient_id = self.load_patient_data(patient_dir)
            
            # Preprocess
            normalized_data, processed_mask = self.preprocess_data(data_dict, patient_id)
            
            # Generate segmentation
            segmentation, probabilities = self.predict_segmentation(normalized_data, patient_id)
            
            # Save final prediction
            prediction_path = self.save_final_prediction(
                segmentation, patient_id, headers['t1']
            )
            
            # Generate report
            processing_time = (datetime.now() - start_time).total_seconds()
            report = self.generate_summary_report(patient_id, segmentation, processing_time)
            
            logger.info(f"Successfully processed {patient_id} in {processing_time:.2f} seconds")
            
            return {
                'patient_id': patient_id,
                'success': True,
                'prediction_path': str(prediction_path),
                'processing_time': processing_time,
                'report': report
            }
            
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
        
        # Load model
        self.load_model()
        
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
    parser = argparse.ArgumentParser(description='Simplified Brain Tumor Segmentation Inference')
    parser.add_argument('--test_dir', default='test_dir', 
                       help='Directory containing test NIfTI files')
    parser.add_argument('--output_dir', default='test_results_simple',
                       help='Directory to save results')
    parser.add_argument('--model_weights', default='weights/weights_1000instances.h5',
                       help='Path to model weights')
    parser.add_argument('--no_intermediate', action='store_true',
                       help='Skip saving intermediate results')
    
    args = parser.parse_args()
    
    # Create tester
    tester = SimpleBrainTumorSegmentationTester(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_weights=args.model_weights,
        save_intermediate=not args.no_intermediate
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