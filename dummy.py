
# from utils.models import binary_model


# binary_weights_path="BinaryWeights.hdf5"
# from keras.layers import ELU
# from utils.preprocessing import create_dataset_from_patients_directory, create_binary_dataset_from_dataset, \
#     create_cropped_dataset_from_dataset


# n_channels = 20
# model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())
# model.load_weights(binary_weights_path)

# print("\nCreating Cropped Dataset From NonCropped Dataset using the Binary Model")
# non_cropped_dataset_path="final_data2021/data"
# binary_dataset_path = "/data/aniket/BrainTumorSegmentation/final_data2021/binary"
# cropped_dataset_path="final_data2021/cropped"

# create_cropped_dataset_from_dataset(non_cropped_dataset_path, model, cropped_dataset_path)




from utils.dataset import MRIDataset
non_cropped_dataset_path="/data/aniket/BrainTumorSegmentation/final_data2021/data"
binary_dataset_path = "/data/aniket/BrainTumorSegmentation/final_data2021/binary"
cropped_dataset_path="/data/aniket/BrainTumorSegmentation/final_data2021/cropped"

MRIDataset(non_cropped_dataset_path=non_cropped_dataset_path,
                      binary_dataset_path=binary_dataset_path,
                      cropped_dataset_path=cropped_dataset_path)
print("success.")