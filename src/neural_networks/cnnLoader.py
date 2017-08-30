from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils


def MNIST_CNN_Model():
    return load_model('../data/mnist_cnn_model.h5')

def MNIST_CNN_sigmoid_Model():
    return load_model('../data/mnist_small_cnn_model_sigmoid.h5')

def MNIST_CNN_simple_Model():
    return load_model('../data/mnist_simple_cnn_model.h5')

def MNIST_CNN_simple_test():
    load_model('mnist_simple_cnn_model.h5')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)

    scores = model.evaluate(X_train, y_train, verbose=0)
    print("Large CNN Error: %.2f%%" % (100 - scores[1] * 100))

# MNIST_CNN_simple_test()