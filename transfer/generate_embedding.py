import numpy as np
import os
import PIL
import csv
import argparse
from keras.preprocessing import image
from util import load_npy, save_csv, save_npy
from TransferModel import TransferModel
from progress.bar import Bar
from progress.spinner import Spinner

def dataset_csv_transferlearning(csv_file, model):
    with open(csv_file, 'r') as source:
        all_content = csv.reader(source)
        dataset = list(all_content)
    data = None
    all_labels = []
    bar = Bar('Generating embeddings', max=len(dataset))
    for row in dataset:
        all_labels.append(row[0])
        img = image.load_img(row[1], target_size=model.input_shape)
        data_features = model.get_embedding(img)
        if data is None:
            data = np.array([]).reshape((0,) + data_features.shape[1:]) 
        data = np.concatenate((data, data_features), axis=0)
        bar.next()
    bar.finish()
    labels = np.array(all_labels)
    labels = labels.reshape((labels.shape[0], 1))
    print('Data type {}, Labels type {}'.format(data.dtype, labels.dtype))
    return data, labels

def dataset_npy_transferlearning(original_data_path, original_labels_path, height, width, channels, model):
    data, labels = load_npy(original_data_path, original_labels_path)
    shape = (height, width, channels)
    new_data = None
    bar = Bar('Preprocessing dataset', max=len(labels))
    for img_array in data:
        data_orig = img_array.reshape(shape)
        img = image.array_to_img(data_orig, data_format='channels_last')
        img = img.resize(model.input_shape, resample=PIL.Image.NEAREST)
        img = model.preprocess_input(img)
        if new_data is None:
            new_data = np.array([], dtype=data[0].dtype).reshape((0,) + img.shape[1:]) 
        new_data = np.concatenate((new_data, img), axis=0)
        bar.next()
    bar.finish()
    print("Generating embeddings using {} model".format(model.model_name))
    emb_data = model.get_embedding_batch(new_data)
    labels = labels.reshape((labels.shape[0], 1))
    return emb_data, labels

# def dataset_npy_transferlearning(original_data_path, original_labels_path, height, width, channels, model):
#     data, labels = load_npy(original_data_path, original_labels_path)
#     shape = (height, width, channels)
#     new_data = None
#     bar = Bar('Generating embeddings', max=len(labels))
#     for img_array in data:
#         data_orig = img_array.reshape(shape)
#         img = image.array_to_img(data_orig, data_format='channels_last')
#         img = img.resize(model.input_shape, resample=PIL.Image.NEAREST)
#         data_features = model.get_embedding(img)
#         if new_data is None:
#             new_data = np.array([], dtype=data[0].dtype).reshape(0, data_features.shape[1])
#         new_data = np.concatenate((new_data, data_features), axis=0)
#         bar.next()
#     bar.finish()
#     labels = labels.reshape((labels.shape[0], 1))
#     return new_data, labels

def main():
    """
    Script to generate csv with image embeddings using pre-trained models given .npy files (data and labels) or a 
    csv file containing first column: <img_labels>, second column: <path_to_images>
    """
    description = "Generate and save image embeddings using pre-trained models"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-csv',
                        '--csv_file',
                        default=None,
                        type=str, help='csv file (default=None)')
    
    parser.add_argument('-data',
                        '--npy_data',
                        default=None,
                        type=str, help='npy data file (default=None)')  
    parser.add_argument('-labels',
                        '--npy_labels',
                        default=None,
                        type=str, help='npy label file (default=None)') 
    parser.add_argument('-dir',
                        '--save_folder_path',
                        type=str, 
                        default=os.getcwd(),
                        help='path to csv files to be saved (default= current working directory)')
    parser.add_argument('-n',
                        '--save_name',
                        type=str, nargs='?', 
                        default="transferfeature", 
                        help='name of new csv or npy file(s) (default="transferfeature")')  
    model_list = """trained_models: VGG16,
                                    VGG19,
                                    ResNet50,
                                    MobileNet,
                                    Xception,
                                    InceptionV3,
                                    InceptionResNetV2"""
    parser.add_argument("-tm",
                        "--trained_model",
                       type=str, 
                       default='VGG16', 
                       help=model_list + "(default=VGG16)") 
    parser.add_argument("-to-csv",
                        "--save_csv",
                       action="store_true", 
                       default=False, 
                       help="save resulting dataset as a csv file, default is saving as npy files (data and labels separately) (default=False)")
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
    parser.add_argument("-weights",
                        "--nn_weights",
                        type=str,
                        default='imagenet',
                        help="load model with trained weights; None to not load model with pre-trained weights (default=imagenet)")

    user_args = parser.parse_args()

    err_input_msg = "Please provide an input file, if .npy please provide a file for data and another for labels"
    assert user_args.csv_file is not None or ((user_args.npy_data is not None) and (user_args.npy_labels is not None)), err_input_msg

    model = TransferModel(model=user_args.trained_model, weights=user_args.nn_weights)
    
    if (user_args.csv_file is not None):
        data, labels = dataset_csv_transferlearning(user_args.csv_file, model)
    else:
        data, labels = dataset_npy_transferlearning(user_args.npy_data,
                                                    user_args.npy_labels,
                                                    user_args.image_height,
                                                    user_args.image_width,
                                                    user_args.image_channels,
                                                    model)
    if (user_args.save_csv):
        save_csv(user_args.save_name,
                 user_args.save_folder_path,
                 data,
                 labels)
    else:
        save_npy(user_args.save_name,
                 user_args.save_folder_path,
                 data,
                 labels)

if __name__ == '__main__':
    main()

