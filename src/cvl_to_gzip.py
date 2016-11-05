import cPickle
import gzip
import shutil
import pprint
from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np


def readFilesName(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def arrayOfPngFile(file):
    im = Image.open(join("../data/cvl.str/", file), 'r')
    pixel_values = np.array(np.asarray(list(im.getdata())), dtype=np.uint8)

    return [pixel[0] / 256.0 for pixel in pixel_values], im.size

def convert():
    fileList = readFilesName("../data/cvl.str")

    cvlstr = ([arrayOfPngFile(file) for file in fileList],
            [file.split("-",1)[0] for file in fileList])

    output = open('../data/cvl.str.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    cPickle.dump(cvlstr, output)

    output.close()

    with open('../data/cvl.str.pkl', 'rb') as f_in, gzip.open('../data/cvl.str.pkl.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def readAndPrint():
    pkl_file = open('../data/data.pkl', 'rb')

    data1, data2, data3= cPickle.load(pkl_file)
    pprint.pprint(data1)
    pprint.pprint(data2)
    pprint.pprint(data3)

    data2 = cPickle.load(pkl_file)
    pprint.pprint(data2)

    pkl_file.close()

def readMnistAndPrint():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    pprint.pprint(training_data)
    pprint.pprint(validation_data)
    pprint.pprint(test_data)


# readMnistAndPrint()

convert()
#readAndPrint()