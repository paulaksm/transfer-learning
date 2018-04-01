import argparse
from util import load_npy, save_npy
import os

def dataset(original_data_path, original_labels_path):
    data, labels = load_npy(original_data_path, original_labels_path)
    new_data = []
    new_labels = []
    for i in range(100):
        new_data.append(data[i])
        new_labels.append(labels[i])
    data = np.array(new_data)
    labels = np.array(new_labels)
    return data, labels

def main():
    """
    Script to generate image embeddings using pre-trained models given .npy files (data and labels)
    """
    description = "Generate and save image embeddings using pre-trained models"
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('npy_data',
                        type=str, help='npy data file')  
    parser.add_argument('npy_label',
                        type=str, help='npy label file')  
 
    user_args = parser.parse_args()
    default_model = TransferModel(model='VGG16')
    data, labels = dataset_transferlearning(user_args.original_data_path,
                                            user_args.original_labels_path)

    save_npy('toy',
             os.getcwd(),
             data,
             labels)

if __name__ == '__main__':
    main()