# ---Libraries---
# Standard library
import math

# Third-party libraries
import cv2
import numpy as np

from scipy import stats
from skimage import transform as tf
from sklearn.preprocessing import normalize


# Private libraries

# Return the image after Gaussian filtering and Otsu's thresholding
def image_binarization(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th3


# Return the histogram of image by axis, the image can be rotated by angle.
def image_histogram(img, rotationAngle=0, axis=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if rotationAngle != 0:
        edges = rotate_image_by_angle(edges, rotationAngle)

    pixel_matrix = np.array(np.asarray(list(edges)), dtype=np.uint8)
    return np.sum(pixel_matrix, axis=axis)


def histogram_entropy(hist):
    hist = np.array(hist, dtype=np.float64)
    hist /= np.max(hist)

    return stats.entropy(hist)


# Return the average angle for the
def angle_matrix_average(matrix, start=0, end=-1):
    if end == -1:
        end = matrix.shape[0]
    count = np.sum(matrix[start:end, :], axis=0)
    threshold = min(np.max(count), 8)
    count = stats.threshold(count, threshold)
    if np.sum(count) == 0:
        return -1
    return int(np.average(range(0, 360), weights=count))


def hist_image_by_angle(image, angle):
    # Create Afine transform
    afine_tf = tf.AffineTransform(shear=math.tan(math.radians(angle)))

    # Apply transform to image data
    return tf.warp(image, afine_tf, preserve_range=True)


def rotate_image_by_angle(image, angle):
    # Create Afine transform
    afine_tf = tf.AffineTransform(rotation=math.tan(math.radians(angle)))

    # Apply transform to image data
    return tf.warp(image, afine_tf, preserve_range=True)


def get_angle(x1, y1, x2, y2):
    if x2 == x1: return 90
    return np.rad2deg(math.atan((y2 - y1) / float(x2 - x1)))


# Distance function
def distance(xi, yi, xii, yii):
    sq1 = (xi - xii) * (xi - xii)
    sq2 = (yi - yii) * (yi - yii)
    return math.sqrt(sq1 + sq2)


# Simple function for indicating on line grater then length
# return true if the line is greater from length
def line_length_is_grater_then(length, line):
    [x1, y1, x2, y2] = line
    return distance(x1, y1, x2, y2) > length


# 2D gaussian mask - should give the same result as MATLAB's
# fspecial('gaussian',[shape],[sigma])
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def normalize(image):
    binary = np.array(image, dtype=np.int8)

    binary[image == 0] = 1
    binary[image == 255] = -1

    return binary


def sanitize(img, do_normalize=True):
    image = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret3, sanitize_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if do_normalize:
        sanitize_img = normalize(sanitize_img)

    return sanitize_img


# Tests
# ----------
# - Is matlab_style_gauss2D is close to matlab output
#
# result = matlab_style_gauss2D((6, 6), 0.5)
#
# expected = [[9.1013e-12, 2.7131e-08, 1.4813e-06, 1.4813e-06, 2.7131e-08, 9.1013e-12],
#             [2.7131e-08, 8.0875e-05, 4.4156e-03, 4.4156e-03, 8.0875e-05, 2.7131e-08],
#             [1.4813e-06, 4.4156e-03, 2.4108e-01, 2.4108e-01, 4.4156e-03, 1.4813e-06],
#             [1.4813e-06, 4.4156e-03, 2.4108e-01, 2.4108e-01, 4.4156e-03, 1.4813e-06],
#             [2.7131e-08, 8.0875e-05, 4.4156e-03, 4.4156e-03, 8.0875e-05, 2.7131e-08],
#             [9.1013e-12, 2.7131e-08, 1.4813e-06, 1.4813e-06, 2.7131e-08, 9.1013e-12]]
#
#
# print np.alltrue(np.isclose(result,expected,rtol=1e-04))

#