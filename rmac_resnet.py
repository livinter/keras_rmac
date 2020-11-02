from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.keras.layers import Lambda, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
#from keras_applications import resnet
from tensorflow.keras.applications import resnet

from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import numpy as np
import utils

K.set_image_data_format('channels_first')

SIZE = 512
s_x, s_y, s_c = 224, 224, 3

K.set_image_data_format('channels_first')

def addition(x):
    return K.sum(x, axis=1)


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, SIZE, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):
    # load ResNet101
    resnet101_model = resnet.ResNet101V2(include_top=True, weights='imagenet', input_tensor=None,
                                       input_shape=(3, 224, 224),
                                       pooling=None, classes=1000)
    # Load VGG16
    # vgg16_model = VGG16('', input_shape)
    # vgg16_model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=(3, 224, 224), pooling=None, classes=1000)

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # reshape
    #    xxx = K.permute_dimensions(vgg16_model.layers[-5].output, (0, 3, 1, 2))

    # ROI pooling
    layer_name = resnet101_model.layers[-4].name
    layer_output = resnet101_model.layers[-4].output
    # print("layer name : " + layer_name)
    # print(layer_output)
    # print('layer name : ' + vgg16_model.layers[-5].name)
    # print(vgg16_model.layers[-5].output)
    x = RoiPooling([1], num_rois)([layer_output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(SIZE, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(SIZE,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([resnet101_model.input, in_roi], rmac_norm)

    # Load PCA weights
    # todo pca layer is trained by data ???
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    return model


def check(img, regions, model):
    new_size = (s_y, s_x, 3)
    img.resize(new_size, refcheck=False)
    x = image.img_to_array(img[:, :, ::-1])
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)
    # print('Input data : %s, %s. %s' %(str(x.shape[1]), str(x.shape[2]), str(x.shape[3])))
    # Compute RMAC vector
    # print('Extracting RMAC from image...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    # print('RMAC size: %s' % RMAC.shape[1])
    return RMAC


def load_RMAC():
    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(s_x, s_y)
    regions = rmac_regions(Wmap, Hmap, s_c)
    print('Loading RMAC model...')
    model = rmac((s_c, s_y, s_x), len(regions))
    return regions, model


if __name__ == "__main__":
    # Load sample image
    file = utils.DATA_DIR + 'sample.jpg'
    img = image.load_img(file)

    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    # new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    new_size = (224, 224)
    print('Original size: %s, Resized image: %s' % (str(img.size), str(new_size)))
    img = img.resize(new_size)

    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)

    print('Input data : %s, %s. %s' % (str(x.shape[1]), str(x.shape[2]), str(x.shape[3])))

    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    print('Loading RMAC model...')
    model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))

    # Compute RMAC vector
    print('Extracting RMAC from image...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
    print('RMAC size: %s' % RMAC.shape[1])
    print(RMAC)
    print('Done!')
