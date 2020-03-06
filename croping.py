import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from multiprocessing import Pool

DATA_DIR = "./Training Data Set/"  # The directory where the script looks for dataset
CLASSES = ["1st Step", "Annavillas"]
training_set = []  # list to store traing set


# Looping through each class
for Class in CLASSES:
    path = os.path.join(DATA_DIR, Class)
    class_num = CLASSES.index(Class)
     # Looping through each image
    for img in os.listdir(path):
         # Try catch block to avoid errors cause by broken image or format:
        try:
            arr = cv2.imread(os.path.join(path, img), 0)
            dim = arr.shape
            if dim[0] < dim[1]:
               img_arr = cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
            img_arr = img_arr[0:int(img_arr.shape[1]/4), :]
            cv2.imshow("gg", img_arr)
            cv2.waitKey(0)
            training_set.append([img_arr, class_num])



        except:
                # print("Invalid image type or directory")
            pass

