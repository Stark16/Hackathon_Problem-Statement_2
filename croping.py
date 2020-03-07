import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
DATA_DIR = "./Training Data Set"  # The directory where the script looks for dataset
CLASSES = ["1st Step", "Annavillas", "Archies", "Bata", "Belgian Waffle", "Big Bazaar", "Burger King", "Cold Stone Creamery", "Croma", "Dominos"]
training_set = []

for Class in CLASSES:
    path = os.path.join(DATA_DIR, Class)
    class_num = CLASSES.index(Class)
    for img in os.listdir(path):

        try:
            img = cv2.imread(os.path.join(path, img), 0)

            if img.shape[0] < img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = img[0:int(img.shape[0]/3), :]
            img = cv2.GaussianBlur(img, (3, 3), 0)

            d = pytesseract.image_to_data(img, output_type=Output.DICT)
            words = int()
            #print(d.keys())
            for i in range(len(d['text'])):
                if len(d['text'][i]) > 2:
                    words = i
                    break
            #print(words)
            #print(d['text'][words])
            (x, y, w, h) = (d['left'][words], d['top'][words], d['width'][words], d['height'][words])
            #print(x, y, w, h)
            img = img[y-5:y+h+5, :]
            training_set.append([img, class_num])
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 10)
            #cv2.imshow("ad", img)
            #cv2.waitKey(0)
            print("working")
        except:
            pass
print(len(training_set))
