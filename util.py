import numpy as np

def load_npy(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def reshape_npy(img_array, shape):
    return img_array.reshape(shape)

def save_npy(name, folder_path, data, labels):
    data_name = "{}_data.npy".format(name)
    label_name = "{}_labels.npy".format(name)
    data_path = os.path.join(folder_path, data_name)
    labels_path = os.path.join(folder_path, label_name)
    np.save(data_path, data)
    np.save(labels_path, labels)
