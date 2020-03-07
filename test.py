import tensorflow as tf
import cv2


CLASSES = ["1st Step", "Annavillas", "Archies", "Bata", "Belgian Waffle"]


def load_test_img(path):  # Fucntion to load the image for testing, it takes image path as argument
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return arr

model = tf.keras.models.load_model("CNN-64x3.model")        # Loading the previously saved model.

Output = model.predict([load_test_img("im__29.png")])   # Classifying the image by passing the name of test image

print(CLASSES[int(Output[0][0])])   # Printing the output on terminal