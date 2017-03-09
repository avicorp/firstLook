# ---Libraries---
# Standard library
import os
import sys
import math

# Third-party libraries
import cv2
import numpy as np
import scipy.ndimage as ndimage


# Private libraries
import Dt_gfilters

sys.path.append(os.path.abspath("../"))


# Extract Region Of Interest (ROI) from bank check.
def extract(image):
    roi = image[2:4,2:4]
    return roi



# Test
img_angle = cv2.imread('../../assets/Checks/7.png')
angle = calculate_slant_angle_pp(img_angle)
image2 = utils.rotate_image_by_angle(img_angle, angle)
# fined_lines_in_check(image2, 'test7.png')
[bifs, C] = compute_BIFs.computeBIFs(image2, 0.5)
cv2.imwrite('bif7.png', color_BIFs.bifs_to_color_image(bifs))
obifs = compute_OBIFs.computeOBIFs(image2, 0.5)
cv2.imwrite('Obif7.png',color_BIFs.bifs_to_color_image(obifs))
# cv2.imwrite('test7.png', image2)
print angle == 0 #-0.549
