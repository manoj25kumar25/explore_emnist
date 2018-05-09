from mnist import MNIST
import numpy as np
import os

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

# Read all EMNIST test and train data
mndata = MNIST('C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data')

X_train, y_train = mndata.load('C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data/emnist-letters-train-images-idx3-ubyte', 
							'C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data/emnist-letters-train-labels-idx1-ubyte')


'''
X_train, y_train = mndata.load('C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data/emnist-letters-test-images-idx3-ubyte', 
							'C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data/emnist-letters-test-labels-idx1-ubyte')
'''
# Read mapping of the labels and convert ASCII values to chars
print("Read mapping of the labels and convert ASCII values to chars")
mapping = []
with open('C:/Users/manoj/Documents/py_workspace/ocr/EMNIST/data/emnist-letters-mapping.txt') as f:
	for line in f:
		mapping.append(chr(int(line.split()[1])))
print("Convert data to numpy arrays and normalize images to the interval [0, 1]")
# Convert data to numpy arrays and normalize images to the interval [0, 1]
print(len(X_train))
print(len(y_train))


print("###")
print('697932')
print('697932')
print("###")

print(X_train[0])
i=0
#im = Image.frombytes('L',(28,28),X_train[0])
for temp in X_train:
	temp=np.reshape(temp,(28,28))
	temp=temp*255
	im = Image.fromarray(temp).convert('L')
	
	directory= 'image/'+str(y_train[i])
	if not os.path.exists(directory):
		os.makedirs(directory)
	im.save(directory+"/"+str(i)+'.png')
	i=i+1


X_train = np.array(X_train, 'int16') / 255
y_train = np.array(y_train,'int')


'''
X_test = np.array(X_test[:100]) / 255
y_test = np.array(y_test[:100])
'''



print("Creating np array of feature and labels")
features = X_train
labels = y_train

print(set(y_train))
#print(y_train[:10])

list_hog_fd = []
for feature in features:
	fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("Initiating svc classifier")


clf = LinearSVC()

print("Map fea and label ")
clf.fit(hog_features, labels)

print("saving the model to digit pickel")
joblib.dump(clf, "digits_cls.pkl", compress=3)


#clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
#cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
#print('Mean accuracy: ', cv_scores.mean())
#print('      Std dev: ', cv_scores.std())


'''
# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

#download the data set
print("Getting Mnist dataset for manipulation ")
dataset = datasets.fetch_mldata("MNIST Original")

print("Creating np array of feature and labels")
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')


list_hog_fd = []
for feature in features:
	fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("Initiating svc classifier")


clf = LinearSVC()

print("Map fea and label ")
clf.fit(hog_features, labels)

print("saving the model to digit pickel")
joblib.dump(clf, "digits_cls.pkl", compress=3)

'''
