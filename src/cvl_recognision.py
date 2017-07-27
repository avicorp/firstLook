# ---Libraries---
# Standard library
import os
import sys
import cv2

# import network_loader
# import cvl_loader
# import pprint
import numpy as np
# from matplotlib import pyplot as plt

# from skimage.morphology import skeletonize
# from scipy.misc import imresize
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from scipy import ndimage
from skimage import measure
from numberRecognition import NumberRecognition
from bankCheck import BankCheck

sys.path.append(os.path.abspath("../"))
import utils


global_index = 0

def step_function(x):
    if x > 50 or (1 > x > 0.3):
        return 1.0
    else:
        return 0.0


binarization = np.vectorize(step_function, otypes=[np.int])


def blob(img):

    blobs_log = blob_log(img, max_sigma=60, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(img, max_sigma=30, threshold=.2)
    # blobs_dog[:, 2] = blobs_dog[:, 2]

    blobs_doh = blob_doh(img, max_sigma=30, threshold=.01)

    return [blobs_log, blobs_dog, blobs_doh]


def clean_blob(img, map, size):
    size = np.max(map)/2 + size
    for i in range(np.max(map)):
        blob = map == i
        if np.count_nonzero(blob) < size:
            img[blob] = 0


def segment(hist):
    blob = []
    seg = False
    start = 0
    for idx, val in enumerate(hist):
        if val > 0 and not seg:
            seg = True
            start = idx
        if seg and val < 1:
            seg = False
            end = idx
            if np.sum(hist[start:idx]) > 20:
                blob += [[start, end]]
            # if end - start > 20 and np.sum(hist[start:end]) > 40:
            #     blob = blob + [[start,  start + 3 + np.argmin(hist[start+3:end-3])]]
            #     blob = blob + [[start + 3 + np.argmin(hist[start+3:end-3]),  end]]
            #     print " {0} - {1} sum {2}".format(start, end, np.sum(hist[start:end]))
            # else:
            #   blob = blob + [[start, end]]

    if seg and np.sum(hist[start:len(hist)]) > 20:
        blob += [[start, len(hist)]]
    return blob


def getWindow(marix, seg):
    binari_matrix = (marix)#binarization
    histY = np.sum(binari_matrix[:, seg[0]:seg[1]], axis=1)
    segY = segment(histY)
    # if len(segY) > 1:
    #     print "Number segment:" + segY
    return marix[segY[0][0]:segY[len(segY)-1][1], seg[0]:seg[1]]


def center_content(window, seg, centerX):
    shift = (seg[1] - seg[0]) - centerX * 2
    marginXMat = np.zeros((len(window) + 4, 4))
    shiftMat = np.zeros((len(window) + 4, abs(shift) + 4))
    marginMat = np.zeros((2, seg[1] - seg[0]))
    centerWindow = np.concatenate((marginMat, window, marginMat), axis=0)
    if shift > 0:
        return np.concatenate((shiftMat, centerWindow, marginXMat), axis=1)
    else:
        return np.concatenate((marginXMat, centerWindow, shiftMat), axis=1)

def add_border(img, size):
    marginXMat = np.zeros((img.shape[0] + size * 2, size))
    marginMat = np.zeros((size, img.shape[1]))
    centerWindow = np.concatenate((marginMat, img, marginMat), axis=0)
    return np.concatenate((marginXMat, centerWindow, marginXMat), axis=1)

def add_smart_border(img, size):
    X, Y = 0, 0
    if img.shape[1] > img.shape[0]:
        X = img.shape[1] - img.shape[0]
    else:
        Y = img.shape[0] - img.shape[1]
    marginXMat = np.zeros((img.shape[0] + size*2 + (X/2)*2, size + Y/2))
    marginMat = np.zeros((size + X/2, img.shape[1]))
    centerWindow = np.concatenate((marginMat, img, marginMat), axis=0)
    return np.concatenate((marginXMat, centerWindow, marginXMat), axis=1)

def center_content_by_mass(window):
    return window
    centerOfMass = ndimage.measurements.center_of_mass(window)
    marginX = int(14 - window.shape[0]/2)
    marginY = 14 - int(centerOfMass[1])
    marginXMatStart = np.zeros((window.shape[0], marginY))
    marginXMatEnd = np.zeros((window.shape[0], 28 - window.shape[1] - marginY))
    marginYMatStart = np.zeros((marginX, 28))
    marginYMatEnd = np.zeros(((28 -window.shape[0] -marginX), 28))
    centerWindow = np.concatenate((marginXMatStart, window, marginXMatEnd), axis=1)
    return np.concatenate((marginYMatStart, centerWindow, marginYMatEnd), axis=0)

def cleanSmallBlob(img, blobs):
    for blob in blobs:
        y, x, r = blob
        y = int(y)
        x = int(x)
        if r < 3:
            cv2.rectangle(img, (x - 1, y - 1), (x + 1, y + 1), 0, 1)


def readNumber(number_model, img, number):
    global global_index
    # Create a black image
    # newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img = 255 - img
    map = measure.label(img)
    clean_blob(img, map, 40)

    # blobimg = blob(img)
    # cleanSmallBlob(img, blobimg[0])
    # cleanSmallBlob(img, blobimg[1])
    # cleanSmallBlob(img, blobimg[2])

    newimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img[img != 0] = 1
    hist = np.sum(img, axis=0)
    segX = segment(hist)
    for idx, seg in enumerate(segX):
        # if(seg[1]-seg[0]< 28):
        histY = np.sum(img[:, seg[0]:seg[1]], axis=1)
        segY = segment(histY)
        if len(segY) > 0:
            window = getWindow(img, seg)
            print "number stat "  + global_index.__str__()
            print window.shape[0] / float(window.shape[1])
            print np.sum(window) / float((window.shape[0] * window.shape[1]))
            cv2.rectangle(newimg, (seg[0],segY[0][0]), (seg[1],segY[len(segY) - 1][1]), (0,0,255), 1)
            # center_content(window)
            # centerWindow = center_content_by_mass(window)
            number_model.isNumber(add_border(window, 20))

            cv2.imwrite("Number_Window" + global_index.__str__() + ".png", add_smart_border(window, 20) * 255)
            global_index += 1
            print number_model.fromImage(add_smart_border(window, 20))
        # print seg
    cv2.imwrite("number_seggmented_clean1_" + number.__str__() + ".png", newimg)


def readNumberByBlob(number_model, img, number):
    # Create a black image
    global global_index
    newprint = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img = 255 - img
    map = measure.label(img)
    clean_blob(img, map, 40)

    newprint = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    map = measure.label(img)

    for i in range(1,np.max(map)):
        newimg = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        newimg[map == i] = 1
        hist = np.sum(newimg, axis=0)
        segX = segment(hist)
        if len(segX) == 1:
            histY = np.sum(newimg[:, segX[0][0]:segX[0][1]], axis=1)
            segY = segment(histY)
            if len(segY) == 1:
                window = newimg[segY[0][0]:segY[0][1], segX[0][0]:segX[0][1]]
                print "number stat " + global_index.__str__()
                print window.shape[0] / float(window.shape[1])
                print sum(hist) / float(window.shape[0]*window.shape[1])
                cv2.rectangle(newprint, (segX[0][0], segY[0][0]), (segX[0][1], segY[len(segY) - 1][1]), (0, 0, 255), 1)
                number_model.isNumber(add_smart_border(window, 20))
                cv2.imwrite("Number_Window" + global_index.__str__() + ".png", add_smart_border(window, 20) * 255)
                global_index += 1
                print number_model.fromImage(add_smart_border(window, 20))
    cv2.imwrite("number_seggmented_blob_" + number.__str__() + ".png", newprint)


# cvl_images, cvl_labels = cvl_loader.load_data(5)
# kernel = np.ones((3, 3), np.float32) / 3
# kernel[1][1] = 1

# for i in range(0,3):
#     size = cvl_images[i][1]
#     pixel_values = np.array(cvl_images[i][0])
#     pixel_matrix = pixel_values.reshape((size[1], size[0]))
#     # pixel_matrix = signal.convolve(pixel_matrix, kernel, mode='same')
#
#     binari_matrix = (pixel_matrix)
#     hist = np.sum(binari_matrix, axis=0)
#     segX = segment(hist)
#     number = 0
#     average = 0
#
#     for idx, seg in enumerate(segX):
#         window = getWindow(pixel_matrix, seg)
#         if (window.shape[0] <= 28):
#             centerWindow = center_content_by_mass(window)

number = NumberRecognition()

print "----1----"
check1 = BankCheck('../assets/Checks/1.png')
img = check1.amountField()
readNumber(number, img, 1)
print "----1 By Blob----"
readNumberByBlob(number, img, 1)
print "----2----"
check1 = BankCheck('../assets/Checks/2.png')
img = check1.amountField()
readNumber(number, img, 2)
print "----2 By Blob----"
readNumberByBlob(number, img, 2)
print "----3----"
check1 = BankCheck('../assets/Checks/3.png')
img = check1.amountField()
readNumber(number, img, 3)
print "----3 By Blob----"
readNumberByBlob(number, img, 3)
print "----4----"
check1 = BankCheck('../assets/Checks/4.png')
img = check1.amountField()
readNumber(number, img, 4)
print "----4 By Blob----"
readNumberByBlob(number, img, 4)
print "----5----"
check1 = BankCheck('../assets/Checks/5.png')
img = check1.amountField()
readNumber(number, img, 5)
print "----5 By Blob----"
readNumberByBlob(number, img, 5)
print "----6----"
check1 = BankCheck('../assets/Checks/6.png')
img = check1.amountField()
readNumber(number, img, 6)
print "----6 By Blob----"
readNumberByBlob(number, img, 6)
print "----7----"
check1 = BankCheck('../assets/Checks/7.png')
img = check1.amountField()
readNumber(number, img, 7)
print "----7 By Blob----"
readNumberByBlob(number, img, 7)
print "----8----"
check1 = BankCheck('../assets/Checks/8.png')
img = check1.amountField()
readNumber(number, img, 8)
print "----8 By Blob----"
readNumberByBlob(number, img, 8)



#
# Naive
# for i in range(0, size[0] - 28):
#     if len(set(pixel_matrix[i, 0:28])) != 1:
#         window = pixel_matrix[i:28 + i, 0:28]
#         result = net.feedforward(window.reshape(28*28,1))
#         if np.amax(result)>0.9:
#             pprint.pprint(i)
#             pprint.pprint(np.amax(result))
#             pprint.pprint(np.argmax(result))

# skeletonImage = skeletonize(f(imresize(pixel_matrix[2:17, 2:10], (28, 28))))
# kernel = np.ones((3, 3), np.float32) / 3
# kernel[1][1] = 1
# blurimage = signal.convolve(skeletonImage, kernel, mode='same')
#
# plt.imshow(blurimage)
# result = net.feedforward(blurimage.reshape(28 * 28, 1))
#
# pprint.pprint(np.amax(result))
# pprint.pprint(np.argmax(result))

# im = f(imresize(pixel_matrix[2:17, 2:10], (28, 28)))
#
# plt.imshow(im)
# result = net.feedforward(im.reshape(28 * 28, 1))
#
# pprint.pprint(np.amax(result))
# pprint.pprint(np.argmax(result))
