import network_loader
import neural_networks.cnnLoader as cnnLoader
import numpy as np
import cv2

from enum import Enum
from scipy.misc import imresize

index = 0

def imageFilter(img):
    imageFilterResult = imresize(img * 255, (28, 28))
    # imageFilterResult = cv2.cvtColor(imageFilterResult, cv2.COLOR_BGR2GRAY)
    # imageFilterResult = 255 - imageFilterResult

    global index

    cv2.imwrite("window_" + index.__str__() + ".png", imageFilterResult)
    index += 1

    return imageFilterResult


def numberFilter(img, size, flat=True):
    bigImage = imresize(img[size:28 - size, size:28 - size], (28, 28))

    if flat:
        return bigImage.reshape(28 * 28, 1)
    return bigImage


class NetworkType(Enum):
    simple = 1
    CNN = 2
    CNN_sigmoid = 3
    CNN_simple = 4


class NumberRecognition:

    def __init__(self, networkType):
        self.smallNetwork = cnnLoader.MNIST_CNN_sigmoid_Model()
        self.network_type = networkType
        if networkType == NetworkType.simple:
            self.network = network_loader.load_network()
        elif networkType == NetworkType.CNN:
            self.network = cnnLoader.MNIST_CNN_Model()
        elif networkType == NetworkType.CNN_sigmoid:
            self.network = cnnLoader.MNIST_CNN_sigmoid_Model()
        elif networkType == NetworkType.CNN_simple:
            self.network = cnnLoader.MNIST_CNN_simple_Model()
        else:
            raise ValueError('Select network type! (simple, CNN)')

    def complexFeedForward(self, img):
        complexResult = self.network.feedforward(numberFilter(img, 2)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 3)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 4)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 5)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 6)) * 0.1

        return complexResult

    def predict(self, img):
        filter = imageFilter(img)

        if self.network_type == NetworkType.simple:
            result = self.network.feedforward(filter.reshape(28 * 28, 1))
        else:
            result = self.network.predict(filter.reshape(1, 1, 28, 28))

        return result

    def isNumber(self, img):
        filter = imageFilter(img)
        result = self.smallNetwork.predict(filter.reshape(1, 1, 28, 28))

        # print np.max(result)
        # print np.sum(result)
        # print np.std(result)
        return np.sum(result) - np.max(result) < 0.1

    def fromImage(self, img):
        result = self.predict(img)

        # if not np.sum(result) - np.max(result) < 0.5:
        #     return -1

        return np.argmax(result)