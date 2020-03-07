import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
DATA_DIR = "./Training Data Set/1st Step/2.jpg"  # The directory where the script looks for dataset

img = cv2.imread(DATA_DIR, 0)

if img.shape[0] < img.shape[1]:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = img[0:int(img.shape[0]/3), :]
img = cv2.GaussianBlur(img, (3, 3), 0)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
words = int()
print(d.keys())
for i in range(len(d['text'])):
    if len(d['text'][i]) > 2:
        words = i
        break
print(words)
print(d['text'][words])
(x, y, w, h) = (d['left'][words], d['top'][words], d['width'][words], d['height'][words])
print(x, y, w, h)
img = img[y:y+h, :]

cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 10)
cv2.imshow("ad", img)
cv2.waitKey(0)
