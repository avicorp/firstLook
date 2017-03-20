# ---Libraries---
# Standard library
import os
import sys
import math

# Third-party libraries
import cv2

# Private libraries
import algorithms.slant_angle as slant_angle
import algorithms.check_input_fields as check_input_fields


class BankCheck:
    kind = 'canine'  # class variable shared by all instances

    def __init__(self, path):
        self.checkImage = slant_angle.fix_check(path)

    def inputFields(self):
        return check_input_fields.extract(self.checkImage)

    def cleanInputFields(self):
        return check_input_fields.clean(self.checkImage)

    def saveInputFields(self, name, clean=False):
        if clean:
            cv2.imwrite(name, self.cleanInputFields())
        else:
            cv2.imwrite(name, self.inputFields())


check1 = BankCheck('../assets/Checks/1.png')
check1.saveInputFields('checkInputFields1.png', True)

check1 = BankCheck('../assets/Checks/2.png')
check1.saveInputFields('checkInputFields2.png', True)

check1 = BankCheck('../assets/Checks/3.png')
check1.saveInputFields('checkInputFields3.png', True)

check1 = BankCheck('../assets/Checks/4.png')
check1.saveInputFields('checkInputFields4.png', True)

check1 = BankCheck('../assets/Checks/5.png')
check1.saveInputFields('checkInputFields5.png', True)

check1 = BankCheck('../assets/Checks/6.png')
check1.saveInputFields('checkInputFields6.png', True)

check1 = BankCheck('../assets/Checks/7.png')
check1.saveInputFields('checkInputFields7.png', True)

check1 = BankCheck('../assets/Checks/8.png')
check1.saveInputFields('checkInputFields8.png', True)
