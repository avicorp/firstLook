import cv2


# Return the image after Gaussian filtering and Otsu's thresholding
def image_binarization(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th3
