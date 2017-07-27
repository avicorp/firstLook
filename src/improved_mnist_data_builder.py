#### Libraries
# Standard library
import cPickle
import gzip
import cv2

# Third-party libraries
import numpy as np

# Self service
import mnist_loader
import algorithms.compute_BIFs as compute_BIFs
import algorithms.compute_OBIFs as compute_OBIFs
import algorithms.color_BIFs as color_BIFs


def new_row(training_row):
    training_inputs = np.swapaxes(training_row[0], 1, 0).reshape((1, 28, 28))
    [bifs, C] = compute_BIFs.computeBIFs(training_inputs[0], 0.5)
    obifs = compute_OBIFs.computeOBIFs(training_inputs[0], 0.5)
    training_inputs_bifs = np.reshape(bifs, (1, 784)) / 30.0
    training_inputs_obifs = np.reshape(obifs, (1, 784)) / 30.0
    density = np.count_nonzero(training_inputs) / 784.0
    network_input = np.append(np.append(np.swapaxes(training_row[0], 1, 0)[0], training_inputs_bifs[0]),
                              training_inputs_obifs[0])
    training_results = np.append(training_row[1], [0])
    return network_input, density, training_results

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

new_training_data = [new_row(training_row) for training_row in training_data]

t=t+1