import tarfile
import gzip
import os
import shutil
from pathlib import Path

def extract_nifti_gz_from_tar(tar_path, output_dir=None):
    """
    Extract all .nii.gz files from a tar archive, maintaining folder structure.
    
    Args:
        tar_path (str): Path to the tar file
        output_dir (str, optional): Output directory. If None, uses the tar file's directory
    """
    tar_path = Path(tar_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = tar_path.parent / f"{tar_path.stem}_extracted"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting from: {tar_path}")
    print(f"Output directory: {output_dir}")
    
    # Open the tar file
    with tarfile.open(tar_path, 'r') as tar:
        # Get all members (files and directories)
        members = tar.getmembers()
        
        # Filter for .nii.gz files
        nifti_gz_files = [m for m in members if m.name.endswith('.nii.gz') and m.isfile()]
        
        print(f"Found {len(nifti_gz_files)} .nii.gz files")
        
        for member in nifti_gz_files:
            print(f"Processing: {member.name}")
            
            # Extract the file to a temporary location
            tar.extract(member, path=output_dir)
            
            # Get the full path of the extracted .gz file
            gz_file_path = output_dir / member.name
            
            # Determine the output .nii file path (remove .gz extension)
            nii_file_path = gz_file_path.with_suffix('')
            
            # Decompress the .gz file
            try:
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(nii_file_path, 'wb') as nii_file:
                        shutil.copyfileobj(gz_file, nii_file)
                
                # Remove the .gz file after successful extraction
                gz_file_path.unlink()
                print(f"  Extracted to: {nii_file_path}")
                
            except Exception as e:
                print(f"  Error extracting {gz_file_path}: {e}")
    
    print(f"\nExtraction complete! Files saved to: {output_dir}")

def extract_with_progress(tar_path, output_dir=None):
    """
    Enhanced version with progress tracking.
    """
    tar_path = Path(tar_path)
    
    if output_dir is None:
        output_dir = tar_path.parent / f"{tar_path.stem}_extracted"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning tar file: {tar_path}")
    
    # First pass: count .nii.gz files
    with tarfile.open(tar_path, 'r') as tar:
        nifti_gz_files = [m for m in tar.getmembers() if m.name.endswith('.nii.gz') and m.isfile()]
    
    total_files = len(nifti_gz_files)
    print(f"Found {total_files} .nii.gz files to extract")
    
    if total_files == 0:
        print("No .nii.gz files found in the archive!")
        return
    
    # Second pass: extract and decompress
    with tarfile.open(tar_path, 'r') as tar:
        for i, member in enumerate(nifti_gz_files, 1):
            print(f"[{i}/{total_files}] Processing: {member.name}")
            
            try:
                # Extract the file
                tar.extract(member, path=output_dir)
                gz_file_path = output_dir / member.name
                nii_file_path = gz_file_path.with_suffix('')
                
                # Decompress
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(nii_file_path, 'wb') as nii_file:
                        shutil.copyfileobj(gz_file, nii_file)
                
                # Clean up .gz file
                gz_file_path.unlink()
                print(f"  ✓ Extracted to: {nii_file_path.relative_to(output_dir)}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Extraction complete! All files saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    # Basic usage
    tar_file_path = "/data/aniket/BrainTumorSegmentation/archive-2021/BraTS2021_Training_Data.tar"  # Replace with your tar file path
    
    # Option 1: Extract to default location (tar_filename_extracted)
    # extract_nifti_gz_from_tar(tar_file_path)
    
    # Option 2: Extract to specific directory
    # extract_nifti_gz_from_tar(tar_file_path, "custom_output_directory")
    
    # Option 3: Use the enhanced version with progress tracking
    extract_with_progress(tar_file_path)
    
    # Interactive mode - uncomment to use
    """
    import sys
    if len(sys.argv) > 1:
        tar_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        extract_with_progress(tar_path, output_dir)
    else:
        print("Usage: python script.py <tar_file_path> [output_directory]")
        print("Example: python script.py data.tar extracted_files")
    """