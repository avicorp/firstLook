import mnist_loader
from collections import Counter
import network
import cPickle
import gzip
import shutil
import cv2
import numpy as np

letter_footprint_dictionary = {}


def add_to_dictionary(_):
    global letter_footprint_dictionary
    footprint = Counter((_[0] > 0.5).flat)[1]
    number = np.argmax(_[1])
    if not letter_footprint_dictionary.has_key(number):
        letter_footprint_dictionary[number] = []

    letter_footprint_dictionary[number].append(footprint)


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

for number in training_data:
    add_to_dictionary(number)

# Ratio
average_footprint_by_number = []

for footprint_list in letter_footprint_dictionary.values():
    average_footprint_by_number.append(np.average(footprint_list))

average_footprint_by_number_ratio = []
average_footprint_in_mnist = np.average(average_footprint_by_number)

for average_footprint in average_footprint_by_number:
    average_footprint_by_number_ratio.append(float(average_footprint_in_mnist) / average_footprint)

print average_footprint_by_number_ratio
