import network_loader
import cvl_loader
import pprint
import numpy as np
from matplotlib import pyplot as plt

from skimage.morphology import skeletonize
from scipy.misc import imresize
from scipy import signal

def step_function(x):
    if x > 50 or (1 > x > 0.3):
        return 1.0
    else:
        return 0.0


binarization = np.vectorize(step_function, otypes=[np.int])


def segment(hist):
    blob = []
    seg = False
    start = 0
    for idx, val in enumerate(hist):
        if val > 0.2 and not seg:
            seg = True
            start = idx
        if seg and ((val < 0.2 and np.sum(hist[start:idx]) > 5) or (idx == len(hist) - 1)):
            seg = False
            end = idx
            if end - start > 20 and np.sum(hist[start:end]) > 40:
                blob = blob + [[start,  start + 3 + np.argmin(hist[start+3:end-3])]]
                blob = blob + [[start + 3 + np.argmin(hist[start+3:end-3]),  end]]
                print " {0} - {1} sum {2}".format(start, end, np.sum(hist[start:end]))
            else:
                blob = blob + [[start, end]]

    if seg:
        blob = blob + [[start, len(hist)]]
    return blob


def getWindow(marix, seg):
    binari_matrix = (marix)#binarization
    histY = np.sum(binari_matrix[0:28, seg[0]:seg[1]], axis=1)
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



# convert cvl to 28X28 windows
net = network_loader.load_network()
cvl_images, cvl_labels = cvl_loader.load_data()
kernel = np.ones((3, 3), np.float32) / 3
kernel[1][1] = 1

for i in range(0,40):
    size = cvl_images[i][1]
    pixel_values = np.array(cvl_images[i][0])
    pixel_matrix = pixel_values.reshape((size[1], size[0]))
    # pixel_matrix = signal.convolve(pixel_matrix, kernel, mode='same')

    binari_matrix = (pixel_matrix)#binarization
    hist = np.sum(binari_matrix, axis=0)
    segX = segment(hist)
    number = 0
    average = 0

    for idx, seg in enumerate(segX):
        window = getWindow(pixel_matrix, seg)
        centerWindow = center_content(window, seg, np.average(range(1,seg[1]+1-seg[0]), axis=0,  weights=hist[seg[0]:seg[1]]))

        skeletonImage = ((imresize(centerWindow, (28, 28),  interp='bicubic')))#binarization#skeletonize
        #blurImage = signal.convolve(skeletonImage, kernel, mode='same')

        plt.imshow(centerWindow)
        plt.imshow(skeletonImage)
        result = net.feedforward(skeletonImage.reshape(28 * 28, 1))

        # pprint.pprint(np.amax(result))
        # pprint.pprint(np.argmax(result))
        average = average + np.amax(result)
        number = number * 10 + np.argmax(result)

    print "-----------{0}-------------".format(i)
    print "Label: {0} Result: {1} average: {2}".format(
        cvl_labels[i], number, average/len(segX))

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
