from PIL import Image
import cv2,sys
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(sys.argv[1])
plt.imshow(img)
w,h = img.size
size = 1
img.resize((w/size, h/size), Image.ANTIALIAS).save('small.jpg')
imh = cv2.imread('small.jpg')
gray, out = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.060)
cv2.stylization(img,gray)
cv2.imshow("cartoon",gray)
plt.show()
cv2.waitKey(0)
