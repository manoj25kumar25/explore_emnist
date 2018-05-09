
import sys

import preprocess
import cnn


import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import pickle

text_list=['a',' b',' c',' d',' e',' f',' g',' h',' i',' j',' k',' l',' m',' n',' o',' p',' q',' r',' s',' t',' u',' v',' w',' x',' y',' z','<space>']

def main(train):
	# Load all data
	
	
	cr = cnn.CharRecognizer()

	if train:
		X_train, y_train, X_test, y_test, mapping = preprocess.load_data('data')
		# Train the Convolutional Neural Network
		#cr.train_model(X_train, y_train, epochs=10)
		cr.train_model(X_train, y_train, epochs=2)

		# Save the model to 'emnist-cnn.h5'. It can be loaded afterwards with cr.load_model().
		cr.save_model()
	else:
		# Load a trained model instead of training a new one.
		try:
			print("Loading pre trained model")
			cr.load_model()
		except:
			print('[Error] No trained CNN model found.')

	# We can use some keras' Sequential model methods too, like summary():
	cr.model.summary()
	im = cv2.imread("C:/Users/manoj/Documents/py_workspace/ocr/a.png")
	im = cv2.imread("C:/Users/manoj/Documents/py_workspace/ocr/example2.jpg")
	im = cv2.imread("C:/Users/manoj/Documents/py_workspace/ocr/example3.jpg")
	
	# Convert to grayscale and apply Gaussian filtering
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	
	# Threshold the image
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	
	# Find contours in the image
	im2,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Get rectangles contains each contour
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	
	# For each rectangular region, calculate HOG features and predict
	# the digit using Linear SVM.
	print("Detected : ")
	text=""
	for rect in rects:
		# Draw the rectangles
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
		# Make the rectangular region around the digit
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
		# Resize the image
		if roi.any():
			roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
			roi = cv2.dilate(roi, (3, 3))
			print (type(roi))
			cv2.imshow("Resulting Image with Rectangular ROIs", roi)
			cv2.waitKey()
			# Calculate the HOG features
			roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
			
			roi=roi.flatten()
			print(roi)
			#img_conv2d_arr,mapping=preprocess.load_sngl_img(np.array([roi_hog_fd],'float64'))
			img_conv2d_arr,mapping=preprocess.load_sngl_img(np.array([roi]),'data')
			#nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
			
			print(np.array([roi_hog_fd],'float64').shape)
			
			nbr =  cr.read_text(img_conv2d_arr, mapping)
			preds=nbr
			print(str(nbr))
			vlu=str(nbr)
			vlu=text_list[int(nbr) -1]
			text=text+vlu
			print(vlu+" -- "+str(int(nbr[0])))
			cv2.putText(im, vlu, (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
	print(text)
	cv2.imshow("Resulting Image with Rectangular ROIs", im)
	cv2.waitKey()

	# Use the CNN model to recognize the characters in the test set.
	#preds = cr.read_text(X_test, mapping)
	print(preds)

	# Evaluate how well the CNN model is doing. 
	# If it's not good enough, we can try training it with a higher number of epochs.
	cr.evaluate_model(X_train, y_train)

if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'train':
		main(train=True)
	else:
		main(train=False)
