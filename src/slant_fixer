#### Libraries
# Standard library
import math

# Third-party libraries
# from skimage import io
import cv2
from skimage import transform as tf
import numpy as np
from skimage import measure
from scipy import stats

# Local Utils
import utils


# cvl_images, cvl_labels = cvl_loader.load_data(5)

averagePixels = np.vectorize(np.average, otypes=[np.int])


def fun(pixel):
    return 1 - (np.average(pixel) / 255.0)

def angleMatrixAverage(matrix, start, end):
    count = np.sum(matrix[start:end, :], axis=0)
    threshold = min(np.max(count), 8)
    count = stats.threshold(count, threshold)
    return int(np.average(range(0,360), weights=count))

def histSegment(hist, imgThetas, thresholdStart, thresholdEnd):
    blob = []
    seg = False
    start = 0
    for idx, val in enumerate(hist):
        if val > thresholdStart and not seg:
            seg = True
            start = idx
        if seg and ((val < thresholdEnd and np.sum(hist[start:idx]) > 5) or (idx == len(hist) - 1)):
            seg = False
            end = idx
            segAngle = angleMatrixAverage(imgThetas, start, end)
            blob = blob + [[start, end, segAngle]]

    if seg:
        segAngle = angleMatrixAverage(imgThetas, start, len(hist))
        blob = blob + [[start, len(hist), segAngle]]
    return blob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def hist_image_by_angle(image, angle):
    # Create Afine transform
    afine_tf = tf.AffineTransform(shear=math.tan(math.radians(angle)))

    # Apply transform to image data
    return tf.warp(image, afine_tf, preserve_range=True)


def imgLines(img, maxDegree, minDegree):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 5)
    thetas = np.zeros((img.shape[1], 360))

    for line_s in lines:
        for rho, theta in line_s:
            angle = int(np.rad2deg(theta))
            if (((minDegree < angle < maxDegree) or
                     (minDegree < 180 - angle < maxDegree))):
                a = np.cos(theta)
                b = np.sin(theta)

                x = int((rho - img.shape[0] / 2 * b) / a)

                if 0 < x < img.shape[1]:
                    thetas[x, angle] += 1

                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, int(theta * 255 / np.pi)), 2)

    cv2.imwrite('../data/cvl.str/test.png', img)

    return thetas

def segmentation(hist, thetas):
    return 2

def lable_blobs(fileName):
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

    binariImage = utils.image_binarization(img)
    blobs = binariImage < 100

    return measure.label(blobs)


maxDegree = 24
minDegree = 5
img = cv2.imread('../data/cvl.str/25000-0001-08.png')

pixel_matrix = np.array(np.asarray(list(img)), dtype=np.uint8)

pixel_mat = np.array([np.apply_along_axis(fun, axis=1, arr=pixel_line) for pixel_line in pixel_matrix])

hist = np.sum(pixel_mat, axis=0)

imgThetas = imgLines(img, maxDegree, minDegree)
segments0 = histSegment(hist, imgThetas, 0, 0)
segments1 = histSegment(hist, imgThetas, 0, 0.5)
segments2 = histSegment(hist, imgThetas, 0.5, 1)
segments3 = histSegment(hist, imgThetas, 1, 2)

for idx, seg in enumerate(segments0):
    np.sum(imgThetas[seg[0]:seg[1], :], axis=0)