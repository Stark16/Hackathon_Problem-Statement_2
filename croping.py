import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from multiprocessing import Pool
source_path = './Training Data Set/1st Step/2.jpg'

image = cv2.imread(source_path, 0)

dim = image.shape
print(dim)
if dim[0] < dim[1]:
    print('Rotate')
    new = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("gg", new)
    cv2.waitKey(0)

else :
    print('Dont rotate')