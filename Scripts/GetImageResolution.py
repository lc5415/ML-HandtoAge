import numpy as np
import pandas as pd
from skimage import io
import os

training_labels = pd.read_csv("../boneage-training-dataset.csv")


img_path = "../FULLdata/training/" # "../labelled/train/"

training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir(img_path))
training_labels_indices = list(training_labels_indices)

i = 0
image_size = np.empty([len(training_labels_indices), 2])
for filename in training_labels_indices:
    #load image
    image = io.imread(img_path+str(filename)+'.png')
    #reshape image as done previously and store in X_train ith row
    #X_train[i,:] = image.reshape(1, np.prod(image.shape))
    image_size[i,:] = image.shape
    #update indexing variable i
    i +=1

print(image_size)

out = pd.DataFrame(image_size)
out.to_csv("../Results/ResolutionTrainingImages.csv")