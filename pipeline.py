import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2
import pickle


DATA_DIR = "./Training Data Set/"  # The directory where the script looks for dataset
CLASSES = ["1st Step", "Annavillas", "Archies", "Bata", "Belgian Waffle", "Big Bazaar", "Burger King", "Cold Stone Creamery", "Croma", "Dominos"]
training_set = []  # list to store traing set


def pipeline():
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
                training_set.append([img_arr, class_num])



            except:
                # print("Invalid image type or directory")
                pass

    # Shuffling the set:
    random.shuffle(training_set)
    print(len(training_set))

    x_feature = []
    y_label = []

    # Creating numpy array for features and sticking labels:
    for feature, label in training_set:
        x_feature.append(feature)
        y_label.append(label)
    x_feature = np.array(x_feature)

    # Writing the arrays in binary files:

    file = open("Feature.dat", "wb", )  #
    pickle.dump(x_feature, file)
    file.close()

    file = open("Label.dat", "wb")
    pickle.dump(y_label, file)
    file.close()


pipeline()