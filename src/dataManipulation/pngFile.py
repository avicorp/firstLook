from PIL import Image
import numpy as np

im = Image.open("assets/mycheck.png", 'r')
width, height = im.size
pixel_values = np.array(np.asarray(list(im.getdata())), dtype=np.uint8)

print [pixel[0] / 256.0 for pixel in pixel_values]