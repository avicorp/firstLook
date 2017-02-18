# ---Libraries---
# Standard library
import os
import sys

# Third-party libraries
import cv2
import numpy as np
import scipy.ndimage as nd

# Private libraries
sys.path.append(os.path.abspath("../"))
import utils


# DTGFILTERSBANK Generates the Derivative-of-Gaussian (DtG) kernels for the computation of BIFs.
def DtGfiltersBank(sigma):
    x = np.arange(-5 * sigma, 5 * sigma + 1)

    # Compute the 0,0 order kernel
    G = utils.matlab_style_gauss2D((x.size, x.size), sigma)

    # Compute the 1,0 and 0,1 order kernels
    [Gx, Gy] = np.gradient(G)

    # Compute the 2,0, 1,1 and 0,2 order kernels
    [Gxx, Gxy] = np.gradient(Gx)
    [Gyx, Gyy] = np.gradient(Gy)

    kernels = np.append([G], [Gy.transpose()], 0)
    kernels = np.append(kernels, [Gx.transpose()], 0)
    kernels = np.append(kernels, [Gyy.transpose()], 0)
    kernels = np.append(kernels, [Gxy.transpose()], 0)
    kernels = np.append(kernels, [Gxx.transpose()], 0)

    return kernels


# Tests
# ----------
# - Is DtGfiltersBank is close to matlab output
#
# result = DtGfiltersBank(0.5)
#
# expectedG = [[9.1013e-12, 2.7131e-08, 1.4813e-06, 1.4813e-06, 2.7131e-08, 9.1013e-12],
#              [2.7131e-08, 8.0875e-05, 4.4156e-03, 4.4156e-03, 8.0875e-05, 2.7131e-08],
#              [1.4813e-06, 4.4156e-03, 2.4108e-01, 2.4108e-01, 4.4156e-03, 1.4813e-06],
#              [1.4813e-06, 4.4156e-03, 2.4108e-01, 2.4108e-01, 4.4156e-03, 1.4813e-06],
#              [2.7131e-08, 8.0875e-05, 4.4156e-03, 4.4156e-03, 8.0875e-05, 2.7131e-08],
#              [9.1013e-12, 2.7131e-08, 1.4813e-06, 1.4813e-06, 2.7131e-08, 9.1013e-12]]
#
# expectedGy = [[2.7121e-08, 8.0848e-05, 4.4141e-03, 4.4141e-03, 8.0848e-05, 2.7121e-08],
#               [7.4063e-07, 2.2078e-03, 1.2054e-01, 1.2054e-01, 2.2078e-03, 7.4063e-07],
#               [7.2707e-07, 2.1674e-03, 1.1833e-01, 1.1833e-01, 2.1674e-03, 7.2707e-07],
#               [-7.2707e-07, -2.1674e-03, -1.1833e-01, -1.1833e-01, -2.1674e-03, -7.2707e-07],
#               [-7.4063e-07, -2.2078e-03, -1.2054e-01, -1.2054e-01, -2.2078e-03, -7.4063e-07],
#               [-2.7121e-08, -8.0848e-05, -4.4141e-03, -4.4141e-03, -8.0848e-05, -2.7121e-08]]
#
# expectedGx = [[2.7121e-08, 7.4063e-07, 7.2707e-07, -7.2707e-07, -7.4063e-07, -2.7121e-08],
#               [8.0848e-05, 2.2078e-03, 2.1674e-03, -2.1674e-03, -2.2078e-03, -8.0848e-05],
#               [4.4141e-03, 1.2054e-01, 1.1833e-01, -1.1833e-01, -1.2054e-01, -4.4141e-03],
#               [4.4141e-03, 1.2054e-01, 1.1833e-01, -1.1833e-01, -1.2054e-01, -4.4141e-03],
#               [8.0848e-05, 2.2078e-03, 2.1674e-03, -2.1674e-03, -2.2078e-03, -8.0848e-05],
#               [2.7121e-08, 7.4063e-07, 7.2707e-07, -7.2707e-07, -7.4063e-07, -2.7121e-08]]
#
# expectedGyy = [[7.1351e-07, 2.1270e-03, 1.1613e-01, 1.1613e-01, 2.1270e-03, 7.1351e-07],
#                [3.4998e-07, 1.0433e-03, 5.6960e-02, 5.6960e-02, 1.0433e-03, 3.4998e-07],
#                [-7.3385e-07, -2.1876e-03, -1.1944e-01, -1.1944e-01, -2.1876e-03, -7.3385e-07],
#                [-7.3385e-07, -2.1876e-03, -1.1944e-01, -1.1944e-01, -2.1876e-03, -7.3385e-07],
#                [3.4998e-07, 1.0433e-03, 5.6960e-02, 5.6960e-02, 1.0433e-03, 3.4998e-07],
#                [7.1351e-07, 2.1270e-03, 1.1613e-01, 1.1613e-01, 2.1270e-03, 7.1351e-07]]
#
# expectedGxy = [[8.0821e-05, 2.2071e-03, 2.1666e-03,-2.1666e-03,-2.2071e-03,-8.0821e-05],
#                 [2.2071e-03, 6.0270e-02, 5.9167e-02,-5.9167e-02,-6.0270e-02,-2.2071e-03],
#                 [2.1666e-03, 5.9167e-02, 5.8084e-02,-5.8084e-02,-5.9167e-02,-2.1666e-03],
#                 [-2.1666e-03,-5.9167e-02,-5.8084e-02, 5.8084e-02, 5.9167e-02, 2.1666e-03],
#                 [-2.2071e-03,- 6.0270e-02,-5.9167e-02, 5.9167e-02, 6.0270e-02, 2.2071e-03],
#                 [-8.0821e-05,-2.2071e-03,-2.1666e-03, 2.1666e-03, 2.2071e-03, 8.0821e-05]]
#
# expectedGxx = [[7.1351e-07, 3.4998e-07,-7.3385e-07,-7.3385e-07, 3.4998e-07, 7.1351e-07],
#                 [2.1270e-03, 1.0433e-03,-2.1876e-03,-2.1876e-03, 1.0433e-03, 2.1270e-03],
#                 [1.1613e-01, 5.6960e-02,-1.1944e-01,-1.1944e-01, 5.6960e-02, 1.1613e-01],
#                 [1.1613e-01, 5.6960e-02,-1.1944e-01,-1.1944e-01, 5.6960e-02, 1.1613e-01],
#                 [2.1270e-03, 1.0433e-03,-2.1876e-03,-2.1876e-03, 1.0433e-03, 2.1270e-03],
#                 [7.1351e-07, 3.4998e-07,-7.3385e-07,-7.3385e-07, 3.4998e-07, 7.1351e-07]]
#
# print np.alltrue(np.isclose(result[0], expectedG, rtol=1e-04))
# print np.alltrue(np.isclose(result[1], expectedGy, rtol=1e-04))
# print np.alltrue(np.isclose(result[2], expectedGx, rtol=1e-04))
# print np.alltrue(np.isclose(result[3], expectedGyy, rtol=1e-04))
# print np.alltrue(np.isclose(result[4], expectedGxy, rtol=1e-04))
# print np.alltrue(np.isclose(result[5], expectedGxx, rtol=1e-04))

#
