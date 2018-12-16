import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from skimage.transform import resize

img = Image.open('C:/Users/Gabe/Documents/LearningAI/ImageMatrixClassification/orig.png').convert('RGBA').resize([600,600],Image.ANTIALIAS)
img2 = Image.open('C:/Users/Gabe/Documents/LearningAI/ImageMatrixClassification/sad.jpg').convert('RGBA').resize([600,600],Image.ANTIALIAS)
# record the original shape
'''
shape = arr.shape
shape2 = arr2.shape
'''
#img = resize(img, (600, 600), anti_aliasing=True)
#img2 = resize(img2, (600, 600), anti_aliasing=True)
#Convert image to numpy matrix
arr = np.array(img)
arr2 = np.array(img2)
#Add
arrays = arr * arr2
shape = arrays.shape
# make a 1-dimensional view of arrays
flat_arr = arrays.ravel()
# convert it to a matrix
vector = np.matrix(flat_arr)

print(vector)

# reform a numpy array of the original shape
arrNew = np.asarray(vector).reshape(shape)

# make a PIL image
imgNew = Image.fromarray(arrNew, 'RGBA')
imgNew.show()





# make a 1-dimensional view of arr
#flat_arr = arr.ravel()
#flat_arr2 = arr2.ravel()
# convert it to a matrix
#vector = np.matrix(flat_arr)
#vector2 = np.matrix(flat_arr2)
#Multiply (find dot product)
#newVector = vector.dot(vector2)
#print(newVector)




'''
#Normalize

vector_norm = preprocessing.normalize(img1Shape, norm='l2')
vector2_norm = preprocessing.normalize(img2Shape, norm='l2')

'''
# convert it to a matrix
'''
vector = np.matrix(vector_norm)
vector2 = np.matrix(vector2_norm)
'''


#Create shared vector
'''
vectors = [[vector],[vector2]]
'''
#Normalize that too
#vectors_norm = preprocessing.normalize(vectors, norm='l2')
'''
print(euclidean_distances(vector, vector2))
'''
# reform a numpy array of the original shape
#arrNew = np.asarray(vector).reshape(shape)

# make a PIL image
#imgNew = Image.fromarray(arrNew, 'RGBA')
#imgNew.show()