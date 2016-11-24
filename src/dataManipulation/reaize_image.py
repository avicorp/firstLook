from PIL import Image
from resizeimage import resizeimage
import pprint
import numpy as np

fd_img = open('data/cvl.str/25000-0001-08.png', 'r')
img = Image.open(fd_img)
img = resizeimage.resize_thumbnail(img, [img.size[0], 28])

size = img.size
pixel_values = np.array(list(img.getdata()))
pixel_matrix = pixel_values.reshape((size[1],size[0],3))

for i in range(0, size[0] - 28):
    window = pixel_matrix[i:27+i][0:27]

for i in range(0,27):
    for j in range(0,27):
        if pixel_values[i*size[0]+j][0] != window[i][j][0] : print("false")

img.save('test-image-thumbnail.png', img.format)
fd_img.close()