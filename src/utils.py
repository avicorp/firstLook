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
