import numpy as np
import pandas as pd
import os

def load_npy(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def save_npy(name, folder_path, data, labels):
    data_name = "{}_data.npy".format(name)
    label_name = "{}_labels.npy".format(name)
    data_path = os.path.join(folder_path, data_name)
    labels_path = os.path.join(folder_path, label_name)
    print('Data shape {}, Labels shape {}'.format(data.shape, labels.shape))
    np.save(data_path, data)
    np.save(labels_path, labels)

def save_csv(name, folder_path, data, labels):
    full_path = os.path.join(folder_path, '{}.csv'.format(name))
    print('Data shape {}, Labels shape {}'.format(data.shape, labels.shape))
    data_label = np.column_stack((labels,data))
    df = pd.DataFrame(data_label)
    df.to_csv(full_path, index=None)  
