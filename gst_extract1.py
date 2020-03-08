
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#img = cv2.imread('4.jpg')

#d = pytesseract.image_to_data(img, output_type=Output.DICT)
# print(d.keys())

path = "/home/akshaya/Desktop/bill_test/"

for j in range(1,24):
	lst = []
	img = cv2.imread(os.path.join(path,str(j)+'.jpg'))
	
	d = pytesseract.image_to_data(img, output_type=Output.DICT)
	#cv2.imshow('imm',imgg)

	n_boxes = len(d['text'])
# print(n_boxes)
	for i in range(n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		lst.append([x,y,w,h,d['text'][i]])
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
#print(lst)
	lst2 = np.asarray(lst)
	lst3 = np.unique(lst2, axis=0)

# print(list(lst3[-1]))

	# print(lst3)
	for i in list(lst3):
		a=(i[-1].lower()=='gst')
		b=(i[-1].lower()=='cgst')
	#c=(i[-1].lower()=='grand')
	#d=(i[-1].lower()=='net')
	# print(c)
		if (a) or (b):
		# print(i)
			crp_image = img[int(i[1])-20:int(i[1])+int(i[3])+20, :]
			cv2.imshow('img', crp_image)
			#imgplot = plt.imshow(crp_image)
			#plt.show()
			cv2.waitKey(0)
			custom_config=r' --oem 1 --psm 6 outputbase digits' 
			text=pytesseract.image_to_string(crp_image,config=custom_config)
			print(text)
	
	



cv2.waitKey(0)



