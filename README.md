# explore_emnist

to use it one can directly use already trained data set pkl or h5 tenserflow model / can train there own.

to train i have used emnist letters dataset, and cnn h5 model is 2 gen trained.

to directly use copy all code to a directory and create another directory named data where one need to place letter mapping file 
e.g if one created emnist folder then directory structure will be 

/emnist

/emnist/emnist-cnn.h5

/emnist/data/emnist-letters-mapping.txt

and update the below code in cnn_test.py to pointing to your image file 

cv2.imread("C:/Users/manoj/Documents/py_workspace/ocr/example3.png")

then run the 

python cnn_test.py

it should run. 

To train the model , one can use main.py but have to download emnist data set to /emnist/data folder , can update the nuber of geration by updating epoc argument.

python main.py train

NOTE: the code out here is taken from multiple sources and modified as per requirment 
some sources which might be useful as below
https://github.com/vitords/EMNIST-sandbox
Emnist dataset , unzip all file to newly created data directory named data , read above directory structure for code
http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
