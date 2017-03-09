# ---Libraries---
# Standard library
import os
import sys
import math

# Third-party libraries
import cv2
import numpy as np
import scipy.ndimage as ndimage


# Private libraries
import Dt_gfilters

sys.path.append(os.path.abspath("../"))


# COMPUTEBIFS - Computes basic images features
#
# im            Image used for BIFs computation.
# sigma         Filter scale
# epislon       Amout of the image clasified as flat
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
def computeBIFs(im, sigma=0.5, epsilon=1e-05):
    gray = gray_and_normalize(im)
    # gray = np.array(im, dtype=np.float64) / 255.0

    # Dervative orders list
    orders = [0,1,1,2,2,2]

    # Set jets arrays
    jet = np.zeros((6, gray.shape[0], gray.shape[1]), np.float64)

    # Do the actual computation
    DtGfilters = Dt_gfilters.DtGfiltersBank(sigma)

    for i in range(0, 6):
        jet[i] = ndimage.filters.convolve(gray, DtGfilters[i], mode='constant')*(sigma**orders[i])

    # Compute lambda and mu
    _lambda = 0.5*(np.squeeze(jet[3]) + np.squeeze(jet[5]))
    mu = np.sqrt(0.25 * ((np.squeeze(jet[3]) - np.squeeze(jet[5])) ** 2) + np.squeeze(jet[4]) ** 2)

    # Initialize classifiers array
    c = np.zeros((jet.shape[1], jet.shape[2], 7), np.float64)

    # Compute classifiers
    c[:, :, 0] = epsilon * np.squeeze(jet[0])
    c[:, :, 1] = np.sqrt(np.squeeze(jet[1]) ** 2 + np.squeeze(jet[2]) ** 2)
    c[:, :, 2] = _lambda
    c[:, :, 3] = -_lambda
    c[:, :, 4] = 2 ** (-1 / 2.0) * (mu + _lambda)
    c[:, :, 5] = 2 ** (-1 / 2.0) * (mu - _lambda)
    c[:, :, 6] = mu

    return [np.array(c[:, :].argmax(2)+1, dtype=np.uint8), jet]


def gray_and_normalize(im):
    im = np.array(im, dtype=np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Normalize
    return np.array(gray, dtype=np.float64) / 255.0


# Tests
# 8X8 Matrix
# image = cv2.imread('bifsTestImages/bif-test.png')
# image = [[0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [145.,145.,145.,145.,145.,145.0,145.,145.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.]]
#
# image = [[ 145,0.,0.,0.,0.,0.,145,0.],
#           [ 0.,145,0.,0.,0.,0.,145,  0.],
#           [ 0.,0.,145,0.,0.,0.,145, 0.],
#           [ 0.,0.,0.,145,0.,145,0.,0.],
#           [ 0.,145,0.,0.,145,0.,0.,0.],
#           [ 0.,145,0.,0.,0.,0.,0.,145],
#           [ 145,0.,145,145,145,0.,145,0.],
#           [ 145,0.,0.,0.,0.,145,0.,0.]]
#
# [bifs, C] = computeBIFs(image,0.5)


# print C
# print bifs+1
# print [expectedC, expectedBifs]==computeBIFs(image)

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
# print np.alltrue(np.equal(expected/255.0, gray_and_normalize(image)))


# Check if cv2.filter2D return the same result as imfilter of matlab
# matrix = [[ 145,0.,0.,0.,0.,0.,145,0.],
#           [ 0.,145,0.,0.,0.,0.,145,  0.],
#           [ 0.,0.,145,0.,0.,0.,145, 0.],
#           [ 0.,0.,0.,145,0.,145,0.,0.],
#           [ 0.,145,0.,0.,145,0.,0.,0.],
#           [ 0.,145,0.,0.,0.,0.,0.,145],
#           [ 145,0.,145,145,145,0.,145,0.],
#           [ 145,0.,0.,0.,0.,145,0.,0.]]
#
# matrix = [[0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [ 145,145,145,145,145,145,145,145],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.]]
# matrix = np.squeeze(np.array(matrix, dtype=np.float64))/255.0
# matrix = np.around(matrix, decimals=6)
#
#
# kernel = utils.matlab_style_gauss2D((5, 5), 0.5)
# # kernel = np.around(kernel, decimals=6)
#
# print matrix
# print kernel
# print ndimage.filters.convolve(matrix, kernel, mode='constant')
#
# print kernel
# print result

# matrix = [[ 145,0.,0.,0.,0.,0.,145,0.],
#           [ 0.,145,0.,0.,0.,0.,145,  0.],
#           [ 0.,0.,145,0.,0.,0.,145, 0.],
#           [ 0.,0.,0.,145,0.,145,0.,0.],
#           [ 0.,145,0.,0.,145,0.,0.,0.],
#           [ 0.,145,0.,0.,0.,0.,0.,145],
#           [ 145,0.,145,145,145,0.,145,0.],
#           [ 145,0.,0.,0.,0.,145,0.,0.]]
#
# gray = [[0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [145.,145.,145.,145.,145.,145.0,145.,145.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., 0.]]
# gray = np.squeeze(np.array(gray, dtype=np.float64))/255.0
#
# orders = [0,1,1,2,2,2]
# sigma = 0.5
# # Set jets arrays
# jet = np.zeros((6, gray.shape[0], gray.shape[1]), np.float64)
#
# # Do the actual computation
# DtGfilters = Dt_gfilters.DtGfiltersBank(sigma)
#
# for i in range(0, 6):
#     jet[i] = ndimage.filters.convolve(gray, DtGfilters[i], mode='constant')*(sigma**orders[i])
#     print "----" + i.__str__() + "----"
#     print jet[i]
#     print np.sum(jet[i])

#
# print kernel
# print matrix