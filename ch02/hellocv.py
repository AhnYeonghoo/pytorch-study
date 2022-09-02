import sys
import cv2
import numpy as np

print('hello OpenCV', cv2.__version__)
file_path = 'C:/Users/dksdu/OneDrive/바탕 화면/pytorch_study/ch02/lenna.bmp'

# img = cv2.imread('C:/Users/dksdu/OneDrive/바탕 화면/pytorch_study/ch02/lenna.bmp')
img_array = np.fromfile(file_path, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img is None:
    print("Image load failed!")
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()
