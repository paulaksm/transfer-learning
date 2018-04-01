import argparse
from util import load_npy, save_npy
import os
import numpy as np

def dataset(original_data_path, original_labels_path, samples=100):
    data, labels = load_npy(original_data_path, original_labels_path)
    new_data = []
    new_labels = []
    idx = np.random.randint(low=0, high=len(original_labels_path), size=samples)
    for i in idx:
        new_data.append(data[i])
        new_labels.append(labels[i])
    data = np.array(new_data)
    labels = np.array(new_labels)
    return data, labels

def main():
    """
    Script to generate a toy dataset given an npy (data and labels) input files
    """
    description = "Generate a toy dataset given an npy (data and labels) input files"
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('npy_data',
                        type=str, help='npy data file')  
    parser.add_argument('npy_labels',
                        type=str, help='npy label file')  
 
    user_args = parser.parse_args()
    data, labels = dataset(user_args.npy_data,
                           user_args.npy_labels)

    save_npy('toy',
             os.getcwd(),
             data,
             labels)

if __name__ == '__main__':
    main()
