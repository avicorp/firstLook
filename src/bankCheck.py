# ---Libraries---
# Standard library
import os
import sys
import math
import numpy as np
# Third-party libraries
import cv2
import numpy as np

# Private libraries
import algorithms.slant_angle as slant_angle
import algorithms.check_input_fields as check_input_fields
import numberRecognition


class BankCheck:

    def __init__(self, path):
        self.checkImage = slant_angle.fix_check(path)

    def inputFields(self):
        return check_input_fields.extract(self.checkImage)

    def cleanInputFields(self):
        return check_input_fields.clean(self.checkImage)

    def amountField(self, clean=True):
        return check_input_fields.extractAmount(self.inputFields(), clean)

    def dateField(self):
        return check_input_fields.extractDate(self.inputFields())

    def saveInputFields(self, name, clean=False):
        if clean:
            cv2.imwrite(name, self.cleanInputFields())
        else:
            cv2.imwrite(name, self.inputFields())

    def saveCheck(self, name):
        cv2.imwrite(name, self.checkImage)