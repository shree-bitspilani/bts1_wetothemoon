from utils.dataset_helpers import create_new_dataset
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Get detailed GPU information
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        print("GPU detected successfully!")
else:
    print("No GPU found")

input_dataset_path = '/data/aniket/BrainTumorSegmentation/archive-2021/BraTS2021_Training_Data_extracted'
output_dataset_path = '/data/aniket/BrainTumorSegmentation/final_data2021'

# Already handles training the binary model
create_new_dataset(input_dataset_path, output_dataset_path)




