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

    def __init__(self, path):
        self.checkImage = slant_angle.fix_check(path)

    def inputFields(self):
        return check_input_fields.extract(self.checkImage)

    def cleanInputFields(self):
        return check_input_fields.clean(self.checkImage)

    def amountField(self):
        return check_input_fields.extractAmount(self.inputFields())

    def dateField(self):
        return check_input_fields.extractDate(self.inputFields())

    def saveInputFields(self, name, clean=False):
        if clean:
            cv2.imwrite(name, self.cleanInputFields())
        else:
            cv2.imwrite(name, self.inputFields())


check1 = BankCheck('../assets/Checks/1.png')
check1.saveInputFields('checkInputFields1.png', True)
cv2.imwrite('amountFields1.png', check1.amountField())
# cv2.imwrite('dateFields1.png', check1.dateField())

check1 = BankCheck('../assets/Checks/2.png')
check1.saveInputFields('checkInputFields2.png', True)
cv2.imwrite('amountFields2.png', check1.amountField())
# cv2.imwrite('dateFields2.png', check1.dateField())

check1 = BankCheck('../assets/Checks/3.png')
check1.saveInputFields('checkInputFields3.png', True)
cv2.imwrite('amountFields3.png', check1.amountField())
# cv2.imwrite('dateFields3.png', check1.dateField())

# check1 = BankCheck('../assets/Checks/4.png')
# check1.saveInputFields('checkInputFields4.png', True)
# cv2.imwrite('amountFields4.png', check1.amountField())
# cv2.imwrite('dateFields4.png', check1.dateField())
#
# check1 = BankCheck('../assets/Checks/5.png')
# check1.saveInputFields('checkInputFields5.png', True)
# cv2.imwrite('amountFields5.png', check1.amountField())
# cv2.imwrite('dateFields5.png', check1.dateField())
#
# check1 = BankCheck('../assets/Checks/6.png')
# check1.saveInputFields('checkInputFields6.png', True)
# cv2.imwrite('amountFields6.png', check1.amountField())
# cv2.imwrite('dateFields6.png', check1.dateField())
#
# check1 = BankCheck('../assets/Checks/7.png')
# check1.saveInputFields('checkInputFields7.png', True)
# cv2.imwrite('amountFields7.png', check1.amountField())
# cv2.imwrite('dateFields7.png', check1.dateField())
#
# check1 = BankCheck('../assets/Checks/8.png')
# check1.saveInputFields('checkInputFields8.png', True)
# cv2.imwrite('amountFields8.png', check1.amountField())
# cv2.imwrite('dateFields8.png', check1.dateField())
