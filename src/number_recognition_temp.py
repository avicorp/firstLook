import network_loader
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join
from scipy import ndimage
from scipy.misc import imresize

global_index = 0

def step_function(x):
    if x > 50 or (1 > x > 0.3):
        return 1.0
    else:
        return 0.0


binarization = np.vectorize(step_function, otypes=[np.int])


def read_number_images(dir):
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    images = [cv2.imread(dir + path) for path in onlyfiles[1:]]
    labels = [path[2:3] for path in onlyfiles[1:]]
    return images, labels, onlyfiles[1:]


def number_filter(img, size, flat=True):
    bigImage = imresize(img[size:28-size, size:28-size], (28, 28))
    cv2.imwrite("input_menual/1.png", bigImage)
    if flat:
        return bigImage.reshape(28 * 28, 1)
    return bigImage


def image_parameters(img):
    centerOfMass = ndimage.measurements.center_of_mass(img)
    standardDeviation = ndimage.measurements.standard_deviation(img)
    return centerOfMass, standardDeviation


def complexFeedForward(img, network):
    global global_index
    Result = network.feedforward(number_filter(img, 1)) * 1.0
    complexResult = network.feedforward(number_filter(img, 1)) * 0.1
    complexResult += network.feedforward(number_filter(img, 3)) * 0.1
    complexResult += network.feedforward(number_filter(img, 4)) * 0.1
    complexResult += network.feedforward(number_filter(img, 5)) * 0.1
    complexResult += network.feedforward(number_filter(img, 6)) * 0.1

    cv2.imwrite("Number_Window_temp" + global_index.__str__() + ".png", number_filter(img, 1, False))
    global_index += 1

    return Result


# convert cvl to 28X28 windows
net = network_loader.load_network()
images, labels, fileNames =read_number_images('../assets/number-test-set/')

graes = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
invers = [abs(255 - im) for im in graes]

results = [complexFeedForward(im, net) for im in invers]


parameters = [image_parameters(number_filter(im,1, False)) for im in invers]

gesses = [np.argmax(result) for result in results]
scores = [max(result)  for result in results]

resultSet = zip(labels, gesses, scores, fileNames, results, invers, parameters)

trashold = 0.0

success = [int(gess == int(label) if (score > trashold) else 2) for (label, gess, score, file, result, im, parameter) in resultSet]

np.set_printoptions(suppress=True, precision=3)


print '{0}: {1} ({2})'.format("False", success.count(0), success.count(0) / float(len(success)))
print '{0}: {1} ({2})'.format("True", success.count(1), success.count(1) / float(len(success)))
print '{0}: {1} ({2})'.format("Fail recognition", success.count(2), success.count(2) / float(len(success)))

standard_deviation_false = []
standard_deviation_true = []

print "False Positiv:"
for (label, gess, score, file, result, im, parameter) in resultSet:
    if score > trashold and gess != int(label):
        standard_deviation_false.append(parameter[1])
        # cv2.imwrite("test.png", im)
        print '{0}: {1}  {2}   {3}'.format(file, gess, parameter, np.transpose(result))


print "True Positiv:"
for (label, gess, score, file, result, im, parameter) in resultSet:
    if score > trashold and gess == int(label):
        standard_deviation_true.append(parameter[1])
        print '{0}: {1}  {2}   {3}'.format(file, gess, parameter, np.transpose(result))

print "Fail:"
for (label, gess, score, file, result, im, parameter) in resultSet:
    if score <= trashold:
        # cv2.imwrite("test.png", im)
        print '{0}: {1}  {2}   {3}'.format(file, gess, parameter, np.transpose(result))


standard_deviation_false = np.array(standard_deviation_false)
standard_deviation_true = np.array(standard_deviation_true)

# for (im, label) in testSet:
#     print "__________"
#     print np.argmax(result) == int(label)
#     print np.argmax(result)
#
#     print max(result)
#     print label




# for i in range(0,3):
#     size = cvl_images[i][1]
#     pixel_values = np.array(cvl_images[i][0])
#     pixel_matrix = pixel_values.reshape((size[1], size[0]))
#     # pixel_matrix = signal.convolve(pixel_matrix, kernel, mode='same')
#
#     binari_matrix = (pixel_matrix)#binarization
#     hist = np.sum(binari_matrix, axis=0)
#     segX = segment(hist)
#     number = 0
#     average = 0
#
#     for idx, seg in enumerate(segX):
#         window = getWindow(pixel_matrix, seg)
#         if (window.shape[0] <= 28):
#             centerWindow = center_content_by_mass(window)
#
#             plt.imshow(window)
#             plt.imshow(centerWindow)
#             result = net.feedforward(centerWindow.reshape(28 * 28, 1))
#
#             # pprint.pprint(np.amax(result))
#             # pprint.pprint(np.argmax(result))
#             average = average + np.amax(result)
#             number = number * 10 + np.argmax(result)
#
#     print "-----------{0}-------------".format(i)
#     print "Label: {0} Result: {1} average: {2}".format(
#         cvl_labels[i], number, average/len(segX))
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
