import unittest

# ---Libraries---
# Standard library

# Third-party libraries
import cv2

# Private libraries
import slant_angle


class TestSlantAngle(unittest.TestCase):
    def setUp(self):
        self.img_angle = cv2.imread('../../assets/mycheck.png')

    def test_something(self):
        angle = slant_angle.calculate_slant_angle(self.img_angle, 300, 0, minDegree=0, maxDegree=180)
        self.assertEqual(angle, 89)


if __name__ == '__main__':
    unittest.main()
