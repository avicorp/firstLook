import cPickle
import gzip
import shutil
import pprint
from os import listdir
from os.path import isfile, join

from PIL import Image
from resizeimage import resizeimage
import numpy as np




def readFilesName(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def arrayOfPngFile(file):
    im = Image.open(join("../data/cvl.str/", file), 'r')
    im = resizeimage.resize_height(im, 28)

    pixel_values = np.array(np.asarray(list(im.getdata())), dtype=np.uint8)

    return [1 - (np.average(pixel) / 255.0) for pixel in pixel_values], im.size

def convert(pklSize):
    fileList = readFilesName("../data/cvl.str")

    cvlstr = ([arrayOfPngFile(file) for file in fileList[0:pklSize]],
            [file.split("-",1)[0] for file in fileList[0:pklSize]])

    output = open('../data/cvl' + pklSize.__str__() + '.str.pkl', 'w')

    # Pickle dictionary using protocol 0.
    cPickle.dump(cvlstr, output)

    output.close()

    with open('../data/cvl' + pklSize.__str__() + '.str.pkl', 'r') as f_in, gzip.open('../data/cvl' + pklSize.__str__() + '.str.pkl.gz', 'w') as f_out:
        shutil.copyfileobj(f_in, f_out)

def readAndPrint():
    pkl_file = open('../data/data.pkl', 'r')

    data1, data2, data3= cPickle.load(pkl_file)
    pprint.pprint(data1)
    pprint.pprint(data2)
    pprint.pprint(data3)

    data2 = cPickle.load(pkl_file)
    pprint.pprint(data2)

    pkl_file.close()

def readMnistAndPrint():
    f = gzip.open('../data/mnist.pkl.gz', 'r')
    training_data, validation_data, test_data = cPickle.load(f)
    pprint.pprint(training_data)
    pprint.pprint(validation_data)
    pprint.pprint(test_data)


convert(5)