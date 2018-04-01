import os
import numpy as np
import PIL
import argparse
import TransferModel
from util import load_npy, save_npy

def dataset_transferlearning(original_data_path, original_labels_path, height, width, channels, model):
    data, labels = load_npy(original_data_path, original_labels_path)
    shape = (height, width, channels)
    new_data = []
    for img_array in data:
        data_orig = img_array.reshape(shape)
        img = image.array_to_img(data_orig, data_format='channels_last')
        img = img.resize(model.input_shape, resample=PIL.Image.NEAREST)
        data_features = model.get_embedding(img)
        new_data.append(data_features)
    data = np.array(new_data)
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
    parser.add_argument('npy_folder_path',
                        type=str, 
                        default=os.getcwd(),
                        help='path to npy files to be saved (default= current working directory)')
    parser.add_argument('npy_name',
                        type=str, nargs='?', default="transferfeature", help='name of new npy files (default="transferfeature")')  # noqa
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
    data, labels = dataset_transferlearning(user_args.original_data_path,
                                            user_args.original_labels_path,
                                            user_args.image_height,
                                            user_args.image_width,
                                            user_args.image_channels,
                                            default_model)

    save_npy(user_args.npy_name,
             user_args.npy_folder_path,
             data,
             labels)

if __name__ == '__main__':
    main()




# model = VGG16(weights='imagenet', include_top=True)

# img_path = 'images/left.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# model = Model(model.input, model.layers[-2].output) # remove top FC-layers of new model output
# # model.summary()
# features = model.predict(x)
# # print("feature vector", features.shape)
