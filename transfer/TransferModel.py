import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras import applications as pretrained
from keras.preprocessing import image

class TransferModel(object):

    def __init__(self, model='VGG16', weights='imagenet'):
        self._weights = weights
        if model == 'VGG16':
            self.model = self._set_vgg16() 

        if model == 'VGG19':
            self.model = self._set_vgg19()

        if model == 'ResNet50':
            self.model = self._set_resnet50()

        if model == 'MobileNet':
            self.model = self._set_mobilenet()

        if model == 'Xception':
            self.model = self._set_xception()

        if model == 'InceptionV3':
            self.model = self._set_inception_v3()

        if model == 'InceptionResNetV2':
            self.model = self._set_inception_resnetv2()

    def get_embedding(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_type.preprocess_input(x)
        feature_vector = self.model.predict(x)
        return feature_vector 

    def _set_vgg16(self):
        self.preprocess_type = pretrained.vgg16
        self.input_shape = (224,224)
        self.model = pretrained.vgg16.VGG16(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()

    def _set_vgg19(self):
        self.preprocess_type = pretrained.vgg19
        self.input_shape = (224,224)
        self.model = pretrained.vgg19.VGG19(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()

    def _set_resnet50(self):
        self.preprocess_type = pretrained.resnet50
        self.input_shape = (224,224)
        self.model = pretrained.resnet50.ResNet50(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()


    def _set_mobilenet(self):
        self.preprocess_type = pretrained.mobilenet
        self.input_shape = (224,224)
        self.model = pretrained.mobilenet.MobileNet(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer(remove_top=6)
        
    def _set_xception(self):
        self.preprocess_type = pretrained.xception
        self.input_shape = (299,299)
        self.model = pretrained.xception.Xception(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()

    def _set_inception_v3(self):
        self.preprocess_type = pretrained.inception_v3
        self.input_shape = (299,299)
        self.model = pretrained.inception_v3.InceptionV3(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()

    def _set_inception_resnetv2(self):
        self.preprocess_type = pretrained.inception_resnet_v2
        self.input_shape = (299,299)
        self.model = pretrained.inception_resnet_v2.InceptionResNetV2(weights=self._weights, include_top=True)
        # print(self.model.summary())
        return self._get_last_layer()

    def _get_last_layer(self, remove_top=2):
        return Model(self.model.input, self.model.layers[-remove_top].output)

# def main():
#     tf = TransferModel(model='Xception')
#     img_path = 'images/left.png'
#     img = image.load_img(img_path, target_size=tf.input_shape)
#     # x = image.img_to_array(img)
#     # x = np.expand_dims(x, axis=0)
#     # x = tf.preprocess_type.preprocess_input(x)
#     features = tf.get_embedding(img)
#     print("feature vector", features.shape)

# if __name__ == '__main__':
#     main()
