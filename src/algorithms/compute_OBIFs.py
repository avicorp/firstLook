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
import compute_BIFs
sys.path.append(os.path.abspath("../"))
import utils

np.seterr(divide='ignore', invalid='ignore')

def quantization(a):
    directionAngles = [0, -45, -90, -135, -180, 180, 135, 90, 45]
    return np.argmin(np.abs(np.array(directionAngles) - a))


oBIFsQuantization = np.vectorize(quantization)

# COMPUTEBIFS - Computes basic images features
#
# im            Image used for BIFs computation.
# sigma         Filter scale
# epislon       Amout of the image clasified as flat
#
# ----- Literature References:
# Griffin et al.
# Basic Image Features (BIFs) Arising from Approximate Symmetry Type.
# Proceedings of the 2nd International Conference on Scale Space and Variational Methods in Computer Vision (2009)
#
# Griffin and Lillholm.
# Symmetry sensitivities of derivative-of-Gaussian filters.
# IEEE Trans Pattern Anal Mach Intell (2010) vol. 32 (6) pp. 1072-83
def computeOBIFs(im, sigma=0.5, epsilon=1e-05):
    [bifs, jet] = compute_BIFs.computeBIFs(im, sigma, epsilon)

    obifs = np.zeros(bifs.shape, np.float64)
    obifs[bifs == 1] = 1

    mask = bifs == 2

    slope_gradient = np.rad2deg(np.arctan(jet[1] / jet[0]))
    slope_gradient = oBIFsQuantization(slope_gradient) + 1
    slope_gradient = np.array(slope_gradient, np.uint8)

    slope_gradient[slope_gradient == 6] = 5
    slope_gradient[slope_gradient > 5] -= 1

    obifs[mask] = 1 + slope_gradient[mask]

    gradient = np.arctan((2 * jet[4]) / (jet[5] - jet[4]))
    gradient = oBIFsQuantization(gradient) + 1
    gradient = np.array(gradient, np.uint8)

    gradient[gradient == 5] = 1
    gradient[gradient == 6] = 1
    gradient[gradient == 7] = 2
    gradient[gradient == 8] = 3
    gradient[gradient == 9] = 4

    mask = bifs == 3
    obifs[mask] = 10

    mask = bifs == 4
    obifs[mask] = 11

    mask = bifs == 5
    obifs[mask] = 11 + gradient[mask]

    mask = bifs == 6
    obifs[mask] = 15 + gradient[mask]

    mask = bifs == 7
    obifs[mask] = 19 + gradient[mask]

    return obifs



#  --- Test ---
#
# print quantization(43) == 8
# print oBIFsQuantization(np.array([43,32.999, -133.99999, 180.9999, 179.9999]))


# image = [[ 145,0.,0.,0.,0.,0.,145,0.],
#           [ 0.,145,0.,0.,0.,0.,145,  0.],
#           [ 0.,0.,145,0.,0.,0.,145, 0.],
#           [ 0.,0.,0.,145,0.,145,0.,0.],
#           [ 0.,145,0.,0.,145,0.,0.,0.],
#           [ 0.,145,0.,0.,0.,0.,0.,145],
#           [ 145,0.,145,145,145,0.,145,0.],
#           [ 145,0.,0.,0.,0.,145,0.,0.]]
#
# obifs = computeOBIFs(image,0.5)
#
# print obifs