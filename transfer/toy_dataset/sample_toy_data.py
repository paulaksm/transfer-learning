import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import load_npy, save_npy

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
    parser.add_argument('-s',
                        '--num_samples',
                        default=100,
                        type=int, help='number of samples (default=100)') 
    parser.add_argument('-n',
                        '--name_npy',
                        default='toy',
                        type=str, help='name of new file (default=toy)')
 
    user_args = parser.parse_args()
    data, labels = dataset(user_args.npy_data,
                           user_args.npy_labels,
                           user_args.num_samples)

    save_npy(user_args.name_npy,
             os.getcwd(),
             data,
             labels)

if __name__ == '__main__':
    main()
