# ---Libraries---
# Standard library
import os
import sys
import cv2
import math

# import network_loader
# import cvl_loader
# import pprint
import numpy as np
from collections import Counter
# from matplotlib import pyplot as plt

# from skimage.morphology import skeletonize
# from scipy.misc import imresize
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from scipy import ndimage
from skimage import measure
from numberRecognition import NumberRecognition, NetworkType
from bankCheck import BankCheck
import algorithms.check_input_fields as input_fields

sys.path.append(os.path.abspath("../"))
import utils

AVERAGE_FOOTPRINT_BY_NUMBER_RATIO = [0.65820361900908462, 1.7295678062438236, 0.88601108766304781, 0.92846039414351023,
                                     1.0848037194317557, 1.0267506757531595, 0.95690731388793582, 1.1474684699915507,
                                     0.86899779336921767, 1.0704710642393012]
border = [0.1, 0.15, 0.2, 0.25, 0.275, 0.285, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.4, 0.45, 0.47, 0.5]
trycount = 2
global_index = 0
index = 0
map = [5, -1, 2, 0, -1, 2, 0, -1, 2, -1,
       0, -1, 1, -1, 5, -1, 0, 0, -1, -1,
       -1, 0, -1, 3, -1, 9, 7, 1, 0, 3,
       -1, -1, -1, 3, 0, 9, 7, 1, 1, 2,
       0, -1, 0, 4, 0, -1, 9, -1, 7, 3,
       7, 3, 9, -1, -1, 5, 2, -1, 0, 0,
       -1, -1, 0, -1, 2, -1, 5, -1, 0, 0,
       -1, 2, -1, -1, 0, -1, -1, -1, 1, 2,
       0, 0, -1, 1, 2, 0, 0, -1]

map1 = [-1, 2, 5, 0, 0, -1,
        1, 5, 0, 0, -1, -1,
        3, 0, 9, 7, -1,
        -1, 4, 0, 0, 0, -1,
        7, 3, 9,
        -1, 2, 5, 0, 0, -1,
       -1, 2, 5, 0, 0, -1,
        -1, 1, 2, 0, 0, -1]



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


def clean_blob(img, map, size, by=0):
    size = np.max(map) / 2 + size
    for i in range(np.max(map) + 1):
        blob = map == i
        if np.count_nonzero(blob) < size:
            img[blob] = by


def keep_blob_from_index(img, map, index, by=0):
    for i in range(1, np.max(map) + 1):
        newimg = np.zeros((map.shape[0], map.shape[1]), np.uint8)
        newimg[map == i] = 1
        hist = np.sum(newimg, axis=0)
        if max(hist[index: len(hist)]) == 0:
            img[map == i] = by


def segment(hist, sizeThreshold=0):
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
            if np.sum(hist[start:idx]) > 20 and (end - start) > sizeThreshold:
                blob += [[start, end]]

    if seg and np.sum(hist[start:len(hist)]) > 20:
        blob += [[start, len(hist)]]
    return blob


def getWindow(marix, seg):
    binari_matrix = (marix)  # binarization
    histY = np.sum(binari_matrix[:, seg[0]:seg[1]], axis=1)
    segY = segment(histY)
    # if len(segY) > 1:
    #     print "Number segment:" + segY
    return marix[segY[0][0]:segY[len(segY) - 1][1], seg[0]:seg[1]]


def window(img_map, img):
    img = np.copy(255 - img)
    hist = np.sum(img, axis=0)
    histY = np.sum(img_map, axis=1)
    seg = segment(hist)
    segY = segment(histY, 5)

    if len(seg) > 1 or len(segY) > 1:
        seg2_center = (seg[0][0] + seg[0][1]) / 2.0
        if len(seg) > 1:
            seg2_center = (seg[1][0] + seg[1][1]) / 2.0
        if seg[0][0] < seg2_center < seg[0][1]:
            seg_center = (seg[0][0] + seg[0][1]) / 2.0
            img_center = img.shape[1] / 2.0
            center_ratio = 1 - (math.fabs(seg_center - img_center) / float(img_center))
            return img_map[segY[0][0]:segY[len(segY) - 1][1], seg[0][0]:seg[len(seg) - 1][1]], center_ratio, seg[0][0]

    if len(seg) == 1 and len(segY) == 1:
        seg_center = (seg[0][0] + seg[0][1]) / 2.0
        img_center = img.shape[1] / 2.0
        center_ratio = 1 - (math.fabs(seg_center - img_center) / float(img_center))
        return img_map[segY[0][0]:segY[0][1], seg[0][0]:seg[0][1]], center_ratio, seg[0][0]

    return [], -1, -1


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
    marginXMat = np.zeros((img.shape[0] + size * 2 + (X / 2) * 2, size + Y / 2))
    marginMat = np.zeros((size + X / 2, img.shape[1]))
    centerWindow = np.concatenate((marginMat, img, marginMat), axis=0)
    return np.concatenate((marginXMat, centerWindow, marginXMat), axis=1)


def center_content_by_mass(window):
    return window
    centerOfMass = ndimage.measurements.center_of_mass(window)
    marginX = int(14 - window.shape[0] / 2)
    marginY = 14 - int(centerOfMass[1])
    marginXMatStart = np.zeros((window.shape[0], marginY))
    marginXMatEnd = np.zeros((window.shape[0], 28 - window.shape[1] - marginY))
    marginYMatStart = np.zeros((marginX, 28))
    marginYMatEnd = np.zeros(((28 - window.shape[0] - marginX), 28))
    centerWindow = np.concatenate((marginXMatStart, window, marginXMatEnd), axis=1)
    return np.concatenate((marginYMatStart, centerWindow, marginYMatEnd), axis=0)


def cleanSmallBlob(img, blobs):
    for blob in blobs:
        y, x, r = blob
        y = int(y)
        x = int(x)
        if r < 3:
            cv2.rectangle(img, (x - 1, y - 1), (x + 1, y + 1), 0, 1)


def rectangle_to_size(rectangle):
    return (rectangle[1][0] - rectangle[0][0]) * (rectangle[1][1] - rectangle[0][1])


def only_positive(number):
    return int((number + math.fabs(number)) / 2)


def rectangle_to_Y_center(rectangle):
    return (rectangle[0][1] + rectangle[1][1]) / 2


def rectangle_to_X_center(rectangle):
    return (rectangle[0][0] + rectangle[1][0]) / 2


def containBy(img, r):
    return ((only_positive(r[0][0]),
             only_positive(r[0][1])),
            (r[1][0] if r[1][0] < img.shape[1] else img.shape[1],
             r[1][1] if r[1][1] < img.shape[0] else img.shape[0]))


def recognition(img, r, number_model):
    global global_index

    rfix = containBy(img, r)

    window = img[rfix[0][1]:rfix[1][1], rfix[0][0]:rfix[1][0]]

    # number = number_model.fromImage(add_smart_border(window, int(window.shape[0]*0.33)))
    number = number_model.fromImage(add_smart_border(window, int(window.shape[0] * border[trycount])))
    cv2.imwrite("Number_Smart_Window" + global_index.__str__() + ".png",
                add_smart_border(window, int(window.shape[0] * border[trycount])) * 255)
    global_index += 1

    print number

    # if map[global_index-1] == -1:
    # number = NumberRecognition(NetworkType.CNN_sigmoid)
    # result = number.fromImage(add_smart_border(window, int(window.shape[0]*border[trycount])))
    # return False
    return map[global_index - 1] == number


def fix_rectangle(rectangle, center, size):
    distance1_from_center = center - rectangle[0][1]
    distance2_from_center = rectangle[1][1] - center

    pointYMod1 = only_positive(size / 2 - distance1_from_center)
    pointYMod2 = only_positive(size / 2 - distance2_from_center)

    width = rectangle[1][0] - rectangle[0][0]
    pointXMod = only_positive(size / 2 - width)

    return ((rectangle[0][0] - pointXMod, rectangle[0][1] - pointYMod1),
            (rectangle[1][0] + pointXMod, rectangle[1][1] + pointYMod2))


def amount_segmentation(img, number, number_model):
    img = 255 - img
    map = measure.label(img)
    clean_blob(img, map, 40)

    newprint = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    hist_map = []

    img[img != 0] = 1
    hist = np.sum(img, axis=0)
    segX = segment(hist)
    for idx, seg in enumerate(segX):
        # if(seg[1]-seg[0]< 28):
        histY = np.sum(img[:, seg[0]:seg[1]], axis=1)
        segY = segment(histY)
        hist_map.append((
            (seg[0], segY[0][0]),
            (seg[1], segY[len(segY) - 1][1])))
        # cv2.rectangle(newprint, (seg[0], segY[0][0]), (seg[1], segY[len(segY) - 1][1]), (0, 0, 255), 2)

    map = measure.label(img)

    blob_map = blobMap(map)

    blob_size = [rectangle_to_size(blob) for blob in blob_map]
    hist_size = [rectangle_to_size(hist) for hist in hist_map]

    blob_centerY = [rectangle_to_Y_center(blob) for blob in blob_map]
    hist_centerY = [rectangle_to_Y_center(hist) for hist in hist_map]

    blob_centerX = [rectangle_to_X_center(blob) for blob in blob_map]
    hist_centerX = [rectangle_to_X_center(hist) for hist in hist_map]

    center_blob = int(np.average(blob_centerY))
    center_hist = int(np.average(hist_centerY))

    center_blobX = int(np.average(blob_centerX))
    center_histX = int(np.average(hist_centerX))

    distance = np.fabs(np.array(blob_centerX) - center_blobX) * 2
    distance = 1 - distance / float(distance.max())

    size1 = int(math.sqrt(np.average(blob_size)))
    size2 = int(math.sqrt(np.average(hist_size)))

    cv2.line(newprint, (0, center_blob), (newprint.shape[1], center_blob), (255, 0, 0), 2)
    cv2.line(newprint, (0, center_hist), (newprint.shape[1], center_hist), (255, 255, 0), 2)

    cv2.line(newprint, (center_blobX, 0), (center_blobX, newprint.shape[1]), (255, 0, 0), 2)
    cv2.line(newprint, (center_histX, 0), (center_histX, newprint.shape[1]), (255, 255, 0), 2)

    # fix_blob_map = [fix_rectangle(blob, center_blob, size1) for blob in blob_map]
    # fix_hist_map = [fix_rectangle(hist, center_hist, size2) for hist in hist_map]

    b = zip(blob_map, distance)
    [cv2.rectangle(newprint, r[0], r[1], (0, 255 * d, 255 * d), 2) for r, d in b]

    # [cv2.rectangle(newprint, r[0], r[1], (0, 255, 255), 2) for r in fix_blob_map]
    # [cv2.rectangle(newprint, r[0], r[1], (255, 255, 255), 2) for r in fix_hist_map]

    numbers1 = [recognition(img, r, number_model) for r in blob_map]
    numbers2 = [recognition(img, r, number_model) for r in hist_map]

    print numbers1
    print numbers2

    cv2.imwrite("ammount_seggmented_" + number.__str__() + ".png", newprint)

    return hist_map, blob_map, sum(numbers1) + sum(numbers2)


def blobMap(map, blobCenter=False):
    blob_map = [((-1, -1), (-1, -1), -1)]

    for i in range(1, np.max(map) + 1):
        newimg = np.zeros((map.shape[0], map.shape[1]), np.uint8)
        newimg[map == i] = 1
        hist = np.sum(newimg, axis=0)
        segX = segment(hist)
        if len(segX) == 1:
            histY = np.sum(newimg[:, segX[0][0]:segX[0][1]], axis=1)
            segY = segment(histY)
            if len(segY) == 1:
                r = ((segX[0][0], segY[0][0]),
                     (segX[0][1], segY[len(segY) - 1][1]))
                if blobCenter:
                    r = ((segX[0][0], segY[0][0]),
                         (segX[0][1], segY[len(segY) - 1][1]),
                         (segX[0][0] + segX[0][1]) / 2.0)
                blob_map.append(r)
            else:
                blob_map.append(((-1, -1), (-1, -1), -1))
        else:
            blob_map.append(((-1, -1), (-1, -1), -1))
    return blob_map


def read_Number(number_model, img, number):
    global global_index
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
            print "number stat " + global_index.__str__()
            print window.shape[0] / float(window.shape[1])
            print np.sum(window) / float((window.shape[0] * window.shape[1]))
            cv2.rectangle(newimg, (seg[0], segY[0][0]), (seg[1], segY[len(segY) - 1][1]), (0, 0, 255), 1)
            # center_content(window)
            # centerWindow = center_content_by_mass(window)
            number_model.isNumber(add_border(window, 20))

            cv2.imwrite("Number_Window" + global_index.__str__() + ".png", add_smart_border(window, 20) * 255)
            global_index += 1
            print number_model.fromImage(add_smart_border(window, 20))
            # print seg
    cv2.imwrite("number_seggmented_clean1_" + number.__str__() + ".png", newimg)


def readNumberByWindow(number_model, img, start, windowSize):
    # invers = 255 - img
    step = 1
    results = []

    for i in range(-5,5,step):
        result = readNumber(img, number_model, start + i, windowSize, True)
        # if result[6] != -1:
        results.append(result[6])
        # else:

    return results


def readNumberByBlob(number_model, img, map, number):
    newprint = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    results = []

    for i in range(1, np.max(map) + 1):
        newimg = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        newimg[map == i] = 1
        hist = np.sum(newimg, axis=0)
        segX = segment(hist)
        if len(segX) >= 1:
            result =  readNumber(img, number_model, segX[0][0], segX[0][1] - segX[0][0]) + (i,)
            if result[5] != -1:
                results.append(result)
    return results


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

# number = NumberRecognition(NetworkType.CNN)
# numberSigmoid = NumberRecognition(NetworkType.CNN_sigmoid)
# numberSimple = NumberRecognition(NetworkType.simple)
numberCnnSimple = NumberRecognition(NetworkType.CNN_simple)


def test_set(number):
    global global_index

    global_index = 0
    count = 0
    print "----1----"
    check1 = BankCheck('../assets/Checks/1.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 1, number)
    count += z
    # readNumber(number, img, 1)
    print "----1 By Blob----"
    # readNumberByBlob(number, img, 1)
    print "----2----"
    check1 = BankCheck('../assets/Checks/2.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 2, number)
    count += z
    # readNumber(number, img, 2)
    print "----2 By Blob----"
    # readNumberByBlob(number, img, 2)
    print "----3----"
    check1 = BankCheck('../assets/Checks/3.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 3, number)
    count += z
    # readNumber(number, img, 3)
    print "----3 By Blob----"
    # readNumberByBlob(number, img, 3)
    print "----4----"
    check1 = BankCheck('../assets/Checks/4.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 4, number)
    count += z
    # readNumber(number, img, 4)
    print "----4 By Blob----"
    # readNumberByBlob(number, img, 4)
    print "----5----"
    check1 = BankCheck('../assets/Checks/5.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 5, number)
    count += z
    # readNumber(number, img, 5)
    print "----5 By Blob----"
    # readNumberByBlob(number, img, 5)
    print "----6----"
    check1 = BankCheck('../assets/Checks/6.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 6, number)
    count += z
    # readNumber(number, img, 6)
    print "----6 By Blob----"
    # readNumberByBlob(number, img, 6)
    print "----7----"
    check1 = BankCheck('../assets/Checks/7.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 7, number)
    count += z
    # readNumber(number, img, 7)
    print "----7 By Blob----"
    # readNumberByBlob(number, img, 7)
    print "----8----"
    check1 = BankCheck('../assets/Checks/8.png')
    img = check1.amountField()
    x, y, z = amount_segmentation(img, 8, number)
    count += z
    # readNumber(number, img, 8)
    print "----8 By Blob----"
    # readNumberByBlob(number, img, 8)

    print "____________________________________" + trycount.__str__()
    print count / 87.0


def readNumber(img, number, startIndex, windowSize, catting=True):
    rect_img = np.copy(img[0:img.shape[0], startIndex:startIndex + windowSize])
    invers = 255 - rect_img
    map = measure.label(invers)

    count = Counter(map.flat)
    common3 = count.most_common(3)

    startSeg = 0

    for j in range(1, np.max(map) + 1):
        newimg = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.float16)
        newimg[map == j] = 1

        if j == common3[1][0] and common3[1][1] > 100:
            img_win, center_retio, startSeg = window(newimg, rect_img)
            if (not catting) and len(common3) > 2 and \
                    blobClose(map, common3[1][0], common3[2][0]) and common3[2][1] > 50:
                newimg[map == common3[2][0]] = 1
                img_win, center_retio, startSeg = window(newimg, rect_img)
            if center_retio != -1:
                guess = number.fromImage(add_smart_border(img_win, int(img_win.shape[0] * border[trycount])))
                commonRetio = 1 - common3[2][1] / float(common3[1][1]) if len(common3) > 2 else 1
                footprint = common3[1][1]/float(common3[0][1])
                ratio = img_win.shape[0]/float(img_win.shape[1])
                if not (footprint>0.4 and ratio<0.6):
                    return (startIndex + startSeg, windowSize, footprint, img_win.shape[0]/float(img_win.shape[1]),
                            commonRetio, center_retio, guess)

    return (startIndex + startSeg, windowSize, -1, -1, -1, -1, -1)


def amount_window_old(img, amount, index, number):
    windowSize = 50
    number_array = []
    step = 10

    for i in range(window, amount.shape[1] - window, step):
        rect_img = np.copy(amount[0:amount.shape[0], i:i + windowSize])
        map = 255 - rect_img
        map = measure.label(map)

        count = Counter(map.flat)
        common3 = count.most_common(3)

        for j in range(1, np.max(map) + 1):
            newimg = np.zeros((rect_img.shape[0], rect_img.shape[1]), np.uint8)
            newimg[map == j] = 1

            if j == common3[1][0] and common3[1][1] > 100:
                img_win = window(newimg)
                num = number.fromImage(add_smart_border(img_win, int(newimg.shape[0] * border[trycount])))
                commonRetio = 1 - common3[2][1] / float(common3[1][1])
                number_array.append((i, num, commonRetio))

    print number_array
    # cv2.imwrite("ammount_window_" + index.__str__() + ".png", amount)


def amount_window(img, amount, index, number):
    buffer = 40
    windowSize = 60
    step = 2
    startsPoint = []
    number_array = []
    allnumbers = []
    all_number_array = []
    gess = {}

    img = blobSpacing(img)
    cv2.imwrite("IS itGood?" + index.__str__() + ".png", img)

    for i in range(buffer, amount.shape[1] - buffer, step):
        numbers = []

        # startIndex, num, commonRatio, center_ratio = readNumber(amount, number, i, windowSize)
        # log_result(gess, all_number_array, allnumbers, center_ratio, commonRatio, num, numbers, startIndex)

        startIndex, num, commonRatio, center_ratio = readNumber(img, number, i, windowSize)
        first_commonRatio = commonRatio
        first_centerRatio = center_ratio
        if startIndex != -1:
            startsPoint.append((startIndex / 5) * 5)
        # log_result(gess, all_number_array, allnumbers, center_ratio, commonRatio, num, numbers, startIndex)
        log_result(gess, all_number_array, allnumbers, center_ratio, commonRatio, num, numbers, startIndex)

        # startIndex, num, commonRatio, center_ratio = readNumber(amount, number, i, windowSize, False)
        # log_result(gess, all_number_array, allnumbers, center_ratio, commonRatio, num, numbers, startIndex)

        # startIndex, num, commonRatio, center_ratio = readNumber(img, number, i, windowSize, False)
        # log_result(gess, all_number_array, allnumbers, center_ratio, commonRatio, num, numbers, startIndex)

        # if len(numbers) == 4 and Counter(numbers).most_common(1)[0][1]>2:
        #     num = Counter(numbers).most_common(1)[0][0]
        #     number_array.append([startIndex, num, first_commonRatio, first_centerRatio])

    print Counter(allnumbers).most_common(6)
    print allnumbers
    print number_array
    print all_number_array

    common10 = Counter(startsPoint).most_common(10)

    for number in common10:
        print number[0]
        numberGess = []
        for ges in gess[number[0]]:
            if ges[2] > 0.25 and ges[1] > 0.25:
                numberGess.append(ges[0])
        print Counter(numberGess).most_common(1)

        # gess = {}
        #
        # for x in all_number_array:
        #     if not gess.has_key(x[0]):
        #         gess[x[0]] = []
        #     gess[x[0]].append(x)
        #
        # print gess

        # cv2.imwrite("ammount_window_" + index.__str__() + ".png", amount)


def log_result(gess, all_number_array, allnumbers, center_retio, commonRatio, num, numbers, startIndex):
    if startIndex != -1:
        numbers.append(num)
        allnumbers.append(num)
        all_number_array.append((startIndex, num, commonRatio, center_retio))
        if not gess.has_key((startIndex / 5) * 5):
            gess[(startIndex / 5) * 5] = []
        gess[(startIndex / 5) * 5].append((num, commonRatio, center_retio))
    return all_number_array


def blobClose(map, value1, value2, distance=3):
    whereValue1 = np.argwhere(map == value1)

    for pos in whereValue1:
        whereValue2 = np.argwhere(
            map[pos[0] - distance:pos[0] + distance, pos[1] - distance:pos[1] + distance] == value2)
        if len(whereValue2) > 0:
            return True, pos, whereValue2[0]

    return False, -1, -1


def fillTheGap(img, pos1, pos2):
    new_pos = [pos1[0] + pos2[0] / 2, pos1[1] + pos2[1] / 2]
    img[new_pos[0], new_pos[1]] = 0
    img[new_pos[0], new_pos[1] + 1] = 0
    img[new_pos[0], new_pos[1] - 1] = 0
    img[new_pos[0] - 1, new_pos[1]] = 0
    img[new_pos[0] + 1, new_pos[1]] = 0
    img[new_pos[0], new_pos[1] + 2] = 0
    img[new_pos[0], new_pos[1] - 2] = 0
    img[new_pos[0] - 2, new_pos[1]] = 0
    img[new_pos[0] + 2, new_pos[1]] = 0


def blobSpacing(img):
    newImage = np.ones(img.shape, np.uint8) * 255
    map = np.copy(255 - img)
    map = measure.label(map)
    sizeOfBlob = 40
    clean_blob(map, map, sizeOfBlob)
    keep_blob_from_index(map, map, 40)

    spacingMap = {}
    finalSpacingMap = {}

    blob_map = blobMap(map, True)

    for i in range(1, map.shape[1]):
        left = []
        right = []
        counters = Counter(map[0:map.shape[0], i])

        for count in counters:
            if count > 0:
                center = blob_map[count][2]
                left.append(count) if center <= i else right.append(count)

        for blob_num in left:
            for blob_num_space in right:
                if not spacingMap.has_key((blob_num_space, blob_num)):
                    spacingMap[(blob_num_space, blob_num)] = 0

                    distSTART = abs(blob_map[blob_num_space][0][0] - blob_map[blob_num][0][0])
                    distEND = abs(blob_map[blob_num_space][1][0] - blob_map[blob_num][1][0])
                    if distSTART < 8 or distEND < 8:
                        spacingMap[(blob_num_space, blob_num)] = -1

                        isclose, pos1, pos2 = blobClose(map, blob_num_space, blob_num)
                        if isclose:
                            fillTheGap(newImage, pos1, pos2)

                if spacingMap[(blob_num_space, blob_num)] >= 0:
                    spacingMap[(blob_num_space, blob_num)] += 1

    for spac in spacingMap.items():
        if not finalSpacingMap.has_key(spac[0][0]):
            finalSpacingMap[spac[0][0]] = spac[1]
        finalSpacingMap[spac[0][0]] = max(finalSpacingMap[spac[0][0]], spac[1])

    blobs = range(1, np.max(map) + 1)
    buffer = 0

    for i in range(1, map.shape[1]):
        counters = Counter(map[0:map.shape[0], i])
        spacing = 0
        for count in counters:

            r = blob_map[count]
            if r[2] != -1 and blobs.__contains__(count):
                blobs.remove(count)
                map_temp = map[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                space = 0  # finalSpacingMap[count] if finalSpacingMap.has_key(count) and finalSpacingMap[count] > 0 else 0
                spacing = max(spacing, space)
                window = newImage[r[0][1]:r[1][1], r[0][0] + buffer + space:r[1][0] + buffer + space]
                window[map_temp == count] = 0
        buffer += spacing

    map = np.copy(255 - newImage)
    map = measure.label(map)
    blob_map = blobMap(map, True)

    return newImage, blob_map, map


def clean_similar_simbol(img, imgMap, templet, templetIsSimilar):

    tmp = img[templet[0][1]:templet[1][1], templet[0][0]:templet[1][0]]
    tmp = ndimage.gaussian_filter(tmp, sigma=3)
    templateImage = np.array(tmp, dtype=np.int16)
    templateImage[templateImage <= 160] = 1
    templateImage[templateImage > 210] = 0
    templateImage[templateImage > 160] = -1
    cv2.imwrite("test.png" ,templateImage+1)
    templetCenter = ((templet[0][1] + templet[1][1]) / 2, (templet[0][0] + templet[1][0]) / 2)
    templetIsSimilarCenter = ((templetIsSimilar[0][1] + templetIsSimilar[1][1]) / 2, (templetIsSimilar[0][0] + templetIsSimilar[1][0]) / 2)

    tmpSimilar = img[templetIsSimilar[0][1]:templetIsSimilar[1][1], templetIsSimilar[0][0]:templetIsSimilar[1][0]]
    templateImgSimilar = np.array(tmpSimilar, dtype=np.int16)
    templateImgSimilar[tmpSimilar == 0] = 1

    secondTempletSearchMap = np.zeros(img.shape)
    secondTempletSearchMap[templetIsSimilar[0][1]:templetIsSimilar[1][1],
    templetIsSimilar[0][0]:templetIsSimilar[1][0]] = 1
    centerLast = input_fields.searchTemplateCenterPointIn(imgMap, templateImage, secondTempletSearchMap)
                                                          # threshold=np.sum(abs(templateImgSimilar))/6)

    print centerLast
    print templetIsSimilarCenter

    if centerLast[0] != 0:
        similarImage = img[templet[0][1] + centerLast[0] - templetCenter[0]:templet[1][1] + centerLast[0] - templetCenter[0],
        templet[0][0] + centerLast[1] - templetCenter[1]:templet[1][0] + centerLast[1] - templetCenter[1]]
        similarImage[tmp < 200] = 255
        img[templet[0][1]:templet[1][1], templet[0][0]:templet[1][0]] = 255


def amountFrame(img, firstBlob, lastBlob):
    global index

    if firstBlob[0][0] == -1 or lastBlob[0][0] == -1:
        cv2.imwrite("fix_amount" + index.__str__() + ".png", img)
        index += 1
        return img

    firstBlodSize = (firstBlob[1][1] - firstBlob[0][1]) * (firstBlob[1][0] - firstBlob[0][0])
    lastBlodSize = (lastBlob[1][1] - lastBlob[0][1]) * (lastBlob[1][0] - lastBlob[0][0])

    amountMap = np.array(img, dtype=np.int8)
    amountMap[img == 0] = 1
    amountMap[img > 0] = -1

    if firstBlodSize < lastBlodSize:
        clean_similar_simbol(img, amountMap, firstBlob, lastBlob)
    else:
        clean_similar_simbol(img, amountMap, lastBlob, firstBlob)

    cv2.imwrite("fix_amount" + index.__str__() + ".png", img)
    index += 1
    return img


def test():
    successCount = 0
    index = 0

    for i in range(1, 9):
        print "--------" + i.__str__() + "-----------------"
        check1 = BankCheck('../assets/Checks/' + i.__str__() + '.png')
        img = check1.amountField()
        img, blob_map, map = blobSpacing(img)
        data, full = getFullResults(i, img, map)

        firstBlobNumber = data[0][5]
        lastBlobNumber = data[len(data) - 1][5]
        frame = np.ones(img.shape,  dtype=np.int16) * 255
        frame[map == firstBlobNumber] = 0
        frame[map == lastBlobNumber] = 0
        # fixImage = amountFrame(frame, blob_map[firstBlobNumber], blob_map[lastBlobNumber])
        fixImage, blob_map, map = blobSpacing(img)

        data1, full1 = getFullResults(i, fixImage, map)

        final_Results = []

        for value in full:
            resultOfwindow = readNumberByWindow(numberCnnSimple, fixImage, value[1][0], value[1][4])
            count = Counter(resultOfwindow)

            # value[1][3] == map1[index]
            if count.most_common(1)[0][0] == map1[index]:
                print (value, " Is: ", map1[index])
                successCount += 1
            else:
                print (value, " Need to be: ", map1[index])
                # if (map1[index] == 5):
                print count
                # print readNumber(fixImage, numberSigmoid, value[1][0], value[1][4])
                # print readNumber(fixImage, numberSimple, value[1][0], value[1][4])

            # if ratio[1] < 0.2:
            #     final_Results.append(ratio[0])

            index += 1



        for value in full1:
            resultOfwindow = readNumberByWindow(numberCnnSimple, img, value[1][0], value[1][4])
            count = Counter(resultOfwindow)

            print count
            print value

    print successCount / float(index)
    # print final_Results


    # amount_window(img, amount, i, number)


def getFullResults(id, img, map):
    results = readNumberByBlob(numberCnnSimple, img, map, id)
    values = [(value[0], value[2], value[3], value[6], value[1], value[7]) for value in results]
    # print results
    # print ratioDiff
    dtype = [('index', int), ('footprint', int), ('footprintNormal', float), ('guess', int), ('window', int),
             ('blobNumber', int)]
    data = np.array(values, dtype=dtype)
    data = np.sort(data, order='index')
    average = np.average([value[2] for value in data[1:len(data)]])
    ratios = [abs(result[2] - average) for result in data]
    full = zip(ratios, data)
    return data, full


trycount = 11
for i in range(0,1):
    print border[trycount]
    test()
    trycount += 1
    #
    # trycount = 2
    # test_set(number)
