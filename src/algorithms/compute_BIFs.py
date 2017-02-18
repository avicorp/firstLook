# ---Libraries---
# Standard library
import os
import sys
import math

# Third-party libraries
import cv2
import numpy as np

# Private libraries
import Dt_gfilters

sys.path.append(os.path.abspath("../"))
import utils


# COMPUTEBIFS - Computes basic images features
#
# im            Image used for BIFs computation.
# sigma         Filter scale
# epislon       Amout of the image clasified as flat
# configuration
#
# Return array of two matrix, the fist matrix is the largest classifier and the second matrix is the bif indicator.
#
# ----- Literature References:
# Griffin et al.
# Basic Image Features (BIFs) Arising from Approximate Symmetry Type.
# Proceedings of the 2nd International Conference on Scale Space and Variational Methods in Computer Vision (2009)
#
# Griffin and Lillholm.
# Symmetry sensitivities of derivative-of-Gaussian filters.
# IEEE Trans Pattern Anal Mach Intell (2010) vol. 32 (6) pp. 1072-83
def computeBIFs(im, sigma=0.5, epsilon=1e-05, configuration=0):
    gray = gray_and_normalize(im)

    # Set jets arrays
    jet = np.zeros((6, gray.shape[0], gray.shape[1]), np.float32)

    # Do the actual computation
    DtGfilters = Dt_gfilters.DtGfiltersBank(sigma)

    for i in range(0, 6):
        jet[i] = cv2.filter2D(gray, -1, DtGfilters[i])

    # Compute lambda and mu
    _lambda = 0.5 * np.squeeze(jet[3]) + np.squeeze(jet[5])
    mu = np.sqrt(0.25 * ((np.squeeze(jet[3]) - np.squeeze(jet[5])) ** 2) + np.squeeze(jet[4]) ** 2)

    # Initialize classifiers array
    c = np.zeros((jet.shape[1], jet.shape[2], 7), np.float32)

    # Compute classifiers
    c[:, :, 0] = epsilon * np.squeeze(jet[0])
    c[:, :, 1] = np.sqrt(np.squeeze(jet[1]) ** 2 + np.squeeze(jet[2]) ** 2)
    c[:, :, 2] = _lambda
    c[:, :, 3] = -_lambda
    c[:, :, 4] = 2 ** (-1 / 2) * (mu + _lambda)
    c[:, :, 5] = 2 ** (-1 / 2) * (mu - _lambda)
    c[:, :, 6] = mu

    # cCompute = np.append([epsilon * np.squeeze(jet[0])],
    #                      [np.sqrt(np.squeeze(jet[1]) ** 2 + np.squeeze(jet[2]) ** 2)], 0)
    # cCompute = np.append(cCompute, [_lambda], 0)
    # cCompute = np.append(cCompute, [-_lambda], 0)
    # cCompute = np.append(cCompute, [2 ** (-1 / 2) * (mu + _lambda)], 0)
    # cCompute = np.append(cCompute, [2 ** (-1 / 2) * (mu - _lambda)], 0)
    # cCompute = np.append(cCompute, [mu], 0)
    #
    # for i in range(0, jet.shape[1]):
    #     for j in range(0, jet.shape[2]):
    #         for clas in range(0, 7):
    #             c[i, j, clas] = cCompute[clas, i, j]

    return [c[:, :].max(2), c[:, :].argmax(2)]


def gray_and_normalize(im):
    im = np.array(im, dtype=np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Normalize
    return gray / 255.0


# Tests
# 8X8 Matrix
image = cv2.imread('bifsTestImages/bif-test.png')
expectedC = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    , [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
expectedBifs = [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]
    , [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

[C, bifs] = computeBIFs(image)
print C
print bifs
# print [C, bifs]==computeBIFs(image)


# Expected gray and normalize to return correct values for "bif-test" image
# image = cv2.imread('bifsTestImages/bif-test.png')
# expected = [[ 145,0.,0.,0.,0.,0.,145,0.],
#           [ 0.,145,0.,0.,0.,0.,145,  0.],
#           [ 0.,0.,145,0.,0.,0.,145, 0.],
#           [ 0.,0.,0.,145,0.,145,0.,0.],
#           [ 0.,145,0.,0.,145,0.,0.,0.],
#           [ 0.,145,0.,0.,0.,0.,0.,145],
#           [ 145,0.,145,145,145,0.,145,0.],
#           [ 145,0.,0.,0.,0.,145,0.,0.]]
# expected = np.array(expected, dtype=np.uint8)
#
# print np.alltrue(np.equal(expected/255.0, gray_and_normalize(image)))
