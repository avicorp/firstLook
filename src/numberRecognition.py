import network_loader
import numpy as np
import cv2

from scipy.misc import imresize


def imageFilter(img):
    img[img == 1] = 255
    imageFilterResult = imresize(img, (28, 28))
    # imageFilterResult = cv2.cvtColor(imageFilterResult, cv2.COLOR_BGR2GRAY)
    # imageFilterResult = 255 - imageFilterResult

    return imageFilterResult


def numberFilter(img, size, flat=True):
    bigImage = imresize(img[size:28 - size, size:28 - size], (28, 28))
    if flat:
        return bigImage.reshape(28 * 28, 1)
    return bigImage


class NumberRecognition:

    def __init__(self):
        self.network = network_loader.load_network()

    def complexFeedForward(self, img):
        complexResult = self.network.feedforward(numberFilter(img, 2)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 3)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 4)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 5)) * 0.1
        complexResult += self.network.feedforward(numberFilter(img, 6)) * 0.1

        return complexResult

    def isNumber(self, img):
        filterImg = imageFilter(img)
        result = self.network.feedforward(filterImg.reshape(28 * 28, 1))

        print np.max(result)
        print np.average(result)
        print np.std(result)
        return False

    def fromImage(self, img):
        filterImg = imageFilter(img)
        result = self.network.feedforward(filterImg.reshape(28 * 28, 1))
        cv2.imwrite('sample.png', filterImg)
        # print np.max(result)
        return np.argmax(result)