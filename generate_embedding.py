import numpy as np
import PIL
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from util import load_npy, save_npy, reshape_npy


def get_embedding(img, model):
    img = image.array_to_img(img, data_format='channels_last')
    img = img.resize((224,224), resample=PIL.Image.NEAREST)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature_vector = model.predict(x)
    return feature_vector

def dataset_transferlearning(original_data_path, original_labels_path, height, width, channels, model):
    data, labels = load_npy(original_data_path, original_labels_path)
    shape = (height, width, channels)
    new_data = []
    for img_array in data:
        data_orig = reshape_npy(img_array, shape)
        data_features = get_embedding(data_orig, model)
        new_data.append(data_features)
    data = np.array(new_data)
    return data, labels

def main():
    """
    Script to generate image embeddings using VGG16 given .npy files (data and labels)
    """
    description = "Generate and save image embeddings using VGG16"
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('npy_data',
                        type=str, help='npy data file')  
    parser.add_argument('npy_label',
                        type=str, help='npy label file')  
    parser.add_argument('npy_folder_path',
                        type=str, help='path to npy files to be saved')
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
    default_model = VGG16(weights='imagenet', include_top=True)
    default_model = Model(default_model.input, default_model.layers[-2].output) # remove top FC-layers of new model output
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