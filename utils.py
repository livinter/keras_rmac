import pickle

DATA_DIR = 'keras_rmac/data/'
WEIGHTS_FILE = 'vgg16_weights_th_dim_ordering_th_kernels.h5'
PCA_FILE = 'PCAmatrices.mat'
IMG_SIZE = 1024

import datetime
def ptime(s):
    return str(datetime.timedelta(seconds=int(s+0.5)))


def save_obj(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print("Object saved to %s." % filename)


def load_obj(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    print("Object loaded from %s." % filename)
    return obj


def preprocess_image(x):
    # except BGR input
    # Subtract Mean
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    return x