from train_multiclass import train as train_multiclass_model
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


cropped_dataset_path = '/data/aniket/BrainTumorSegmentation/final_data2021/cropped'
multiclass_model_weights = '/data/aniket/BrainTumorSegmentation/weights/weights_1000instances.h5'
train_multiclass_model(cropped_dataset_path, multiclass_model_weights)