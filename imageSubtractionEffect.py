import numpy as np
from PIL import Image
from sklearn import svm
from skimage.transform import resize

img = Image.open('C:/Users/Gabe/Documents/LearningAI/ImageMatrixClassification/orig.png').convert('RGBA').resize([600,600],Image.ANTIALIAS)
img2 = Image.open('C:/Users/Gabe/Documents/LearningAI/ImageMatrixClassification/sad.jpg').convert('RGBA').resize([600,600],Image.ANTIALIAS)
#Convert Images to arrays
arr = np.array(img)
arr2 = np.array(img2)

#Multiply arrays
arrays = arr2 - arr
shape = arrays.shape

#Make 1D view of arrays
flat_arr = arrays.ravel()

#Convert array to matrix
vector = np.matrix(flat_arr)

#Reform a numpy array of the original shape
arrNew = np.asarray(vector).reshape(shape)

#Make a PIL image
imgNew = Image.fromarray(arrNew, 'RGBA')
imgNew.show()