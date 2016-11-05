#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return a tuple containing ``(image_data, label_data)``
    image_data contain array of tuple (image, size(width, height))"""

    f = gzip.open('../data/cvl.str.pkl.gz', 'rb')
    images_data, lable_data = cPickle.load(f)
    f.close()
    return (images_data, lable_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
