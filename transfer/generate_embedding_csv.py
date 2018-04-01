import numpy as np
from keras.preprocessing import image
import os
import PIL
import csv
import argparse
from util import load_npy, save_csv
from TransferModel import TransferModel
from progress.bar import Bar

def dataset_csv_transferlearning(csv_file):
    with open(csv_file, 'rb') as source:
        all_content = csv.reader(source)
        dataset = list(all_content)
    all_images = []
    all_labels = []
    for row in dataset:
        all_labels.append(row[1])
        img = image.load_img(row[0], target_size=model.input_shape)
        data_features = model.get_embedding(img)
        all_images.append(data_features)

    data = np.array(all_images)
    all_labels = all_labels.reshape((all_labels.shape[0], 1))
    labels = np.array(all_labels)

    return data, labels

def dataset_npy_transferlearning(original_data_path, original_labels_path, height, width, channels, model):
    data, labels = load_npy(original_data_path, original_labels_path)
    shape = (height, width, channels)
    new_data = None
    bar = Bar('Generating embeddings', max=len(labels))
    for img_array in data:
        data_orig = img_array.reshape(shape)
        img = image.array_to_img(data_orig, data_format='channels_last')
        img = img.resize(model.input_shape, resample=PIL.Image.NEAREST)
        data_features = model.get_embedding(img)
        if new_data is None:
            new_data = np.array([], dtype=data[0].dtype).reshape(0, data_features.shape[1])
        new_data = np.concatenate((new_data, data_features), axis=0)
        bar.next()
    bar.finish()
    labels = labels.reshape((labels.shape[0], 1))
    return new_data, labels

def main():
    """
    Script to generate csv with image embeddings using pre-trained models given .npy files (data and labels)
    """
    description = "Generate and save image embeddings using pre-trained models"
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('npy_data',
                        type=str, help='npy data file')  
    parser.add_argument('npy_labels',
                        type=str, help='npy label file') 
    #parser.add_argument('csv_file',
    #                    type=str, nargs='?', default=None, help='csv file with first column: label and second column: image path') 
    parser.add_argument('csv_folder_path',
                        type=str, 
                        default=os.getcwd(),
                        help='path to csv files to be saved (default= current working directory)')
    parser.add_argument('csv_name',
                        type=str, nargs='?', default="transferfeature", help='name of new csv file (default="transferfeature")')  # noqa
    parser.add_argument("-he",
                        "--image_height",
                        type=int,
                        default=90,
                        help="original image height number (default=90)")
    parser.add_argument("-w",
                        "--image_width",
                        type=int,
                        default=160,
                        help="original image width number (default=160)")
    parser.add_argument("-c",
                        "--image_channels",
                        type=int,
                        default=3,
                        help="original image channels (default=3)")

    user_args = parser.parse_args()
    default_model = TransferModel(model='VGG16')
    data, labels = dataset_npy_transferlearning(user_args.npy_data,
                                                user_args.npy_labels,
                                                user_args.image_height,
                                                user_args.image_width,
                                                user_args.image_channels,
                                                default_model)
    save_csv(user_args.csv_name,
             user_args.csv_folder_path,
             data,
             labels)

if __name__ == '__main__':
    main()

