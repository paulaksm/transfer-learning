# Transfer Learning - Image Embeddings Generator

Transfer Learning - Image Embeddings Generator is a simple script that takes advantage of pre-trained deep convolutional neural networks (CNN) on ImageNet to generate a significant feature representation of an input image. 

### Motivation

When dealing with images, no matter how simple the project is, their naive vector representation (width * height * channels) can take an order of magnitude of tens of thousands of features, each! And despite the required memory space, are all the pixels in an image relevant for a classification task? - No!

Transfer Learning for images, more specific Transferring Knowledge of Feature Representations, takes an image and do a forward pass in a CNN trained on another domain for a different task, and returns a low-dimensional representation of the given image. This new representation is expected to be a more significant and smaller vector of the input given that is the result of applying a sequence of pre-learned kernels.

The main goal of this repository is to provide different image embeddings for a quick evaluation of machine learning algorithms trained on different representations of the input data.

### Available pre-trained CNNs

<Add table:>

## Installation

### Requirements
Install all the requirements for Python 3 by running: 

`$ pip3 install -r requirements.txt`

## Usage

```console
$ cd transfer/
$ python generate_embedding.py -h

usage: generate_embedding.py [-h] [-csv CSV_FILE] [-data NPY_DATA]
                             [-labels NPY_LABELS] [-dir SAVE_FOLDER_PATH]
                             [-n [SAVE_NAME]] [-tm TRAINED_MODEL] [-to-csv]
                             [-he IMAGE_HEIGHT] [-w IMAGE_WIDTH]
                             [-c IMAGE_CHANNELS] [-weights NN_WEIGHTS]
                             [-b BATCH_SIZE]

Generate and save image embeddings using pre-trained models

optional arguments:
  -h, --help            show this help message and exit
  -csv CSV_FILE, --csv_file CSV_FILE
                        csv file (default=None)
  -data NPY_DATA, --npy_data NPY_DATA
                        npy data file (default=None)
  -labels NPY_LABELS, --npy_labels NPY_LABELS
                        npy label file (default=None)
  -dir SAVE_FOLDER_PATH, --save_folder_path SAVE_FOLDER_PATH
                        path to csv files to be saved (default= current
                        working directory)
  -n [SAVE_NAME], --save_name [SAVE_NAME]
                        name of new csv or npy file(s)
                        (default="transferfeature")
  -tm TRAINED_MODEL, --trained_model TRAINED_MODEL
                        trained_models: VGG16, VGG19, ResNet50, MobileNet,
                        Xception, InceptionV3,
                        InceptionResNetV2(default=VGG16)
  -to-csv, --save_csv   save resulting dataset as a csv file, default is
                        saving as npy files (data and labels separately)
                        (default=False)
  -he IMAGE_HEIGHT, --image_height IMAGE_HEIGHT
                        original image height number (default=90)
  -w IMAGE_WIDTH, --image_width IMAGE_WIDTH
                        original image width number (default=160)
  -c IMAGE_CHANNELS, --image_channels IMAGE_CHANNELS
                        original image channels (default=3)
  -weights NN_WEIGHTS, --nn_weights NN_WEIGHTS
                        None to load without pre-trained weights 
                        (default=imagenet)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        run predictions on batch of given size (8, 16, 32,
                        64...). If default, images will be passed to the model
                        as they are preprocessed (default=1)
```

## Usage example - Orange

<Add código rodado no brucutu, prints do orange>

## Built with 
* Keras
* TensorFlow

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

### Citation
```
@misc{deeplearning_image_embeddings2018,
    author = {Paula Moraes},
    title = {Transfer Learning - Image Embeddings Generator},
    year = {2018},
    howpublished = {\url{https://github.com/paulaksm/transfer-learning}},
    note = {commit xxxxxxx}
  }
```

## License
[MIT](https://choosealicense.com/licenses/mit/)