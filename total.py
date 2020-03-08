import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import re

imgg = cv2.imread('3.jpg')
img = imgg[list(imgg.shape)[0]//2:, : ]
d = pytesseract.image_to_data(img, output_type=Output.DICT)

keys=list(d.keys())

detect_pattern='total'
detect_invoice='cash'


n_boxes = len(d['text'])
print(n_boxes)
for i in range(n_boxes):
    #if int(d['conf'][i]) >100:
       if re.match(detect_invoice, d['text'][i].lower()):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
       elif re.match(detect_pattern, d['text'][i].lower()):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img1=cv2.rectangle(img, (x, y), (list(img.shape)[1], y + h), (255, 0,255 ), 2)
        crped_image=img[y+10:y+h-5,x: ]
        #cv2.imshow("crpimg",crped_image)   
        d1= pytesseract.image_to_data(crped_image, output_type=Output.DICT)
        custom_config = r'--oem 3 --psm 6 outputbase digits'
        #custom_config=r'--psm 13 --oem 3 outputbase digits '
        print(pytesseract.image_to_string(crped_image, config=custom_config))






#cv2.imshow('img', crped_image)
cv2.imshow('img2',img)


cv2.waitKey(0)



