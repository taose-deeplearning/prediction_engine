
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
from scipy.misc import imresize
import numpy as np
import chainer 
import chainer.functions as F
import pickle
from PIL import Image

# CNN
def predict(X, model):
        X = chainer.Variable(X)
        
        h0 = F.max_pooling_2d(F.relu(model.norm1(model.conv1(X))), 2)
        h1 = F.max_pooling_2d(F.relu(model.norm2(model.conv2(h0))), 2)
        h2 = F.max_pooling_2d(F.relu(model.norm3(model.conv3(h1))), 2)
        h3 = F.relu(model.l1(h2))
        h4 = model.l2(h3)
        prob_y = F.softmax(h4)
        # return probability of class
        return prob_y.data
    
def load_model(model_path):
    with open(model_path, 'rb') as o:
        model = pickle.load(o)
        
        return model

# Exhaustive Search
# Parameters to change
image_path = 'tomato.jpg'
model_object_path = 'model_object.pkl'
model_recog_path = 'model_recog.pkl'

image = np.array(Image.open(image_path))[:, :, ::-1]
model_object = load_model(model_object_path)
model_recog = load_model(model_recog_path)


windows = []
position = []


winW = winH = int(image.shape[0]//4.6)
stepSize = int(winW//4)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:    
            continue
        window_32by32 = imresize(window, (32, 32, 3))/255
        windows.append(window_32by32.flatten())
        position.append([0])
    
windows = np.array(windows)
position = np.array(position)
windows_reshape = windows.reshape((len(windows), 3, 32, 32))

# Object Detection
prob_y = predict(windows.reshape((len(windows), 3, 32, 32)).astype(np.float32), model_object)[:, 1]
mean = np.mean(prob_y[np.where(prob_y>0.5)])
prob_y[np.where(prob_y<mean)] = 0

for i in range(len(prob_y)):
    if prob_y[i]>=mean:
        position[i, 0] = 1
    else: pass

# Object Classifier
index_box = np.where(position[:, 0]==1)
labels = predict(windows[index_box].reshape((len(index_box[0]), 3, 32, 32)).astype(np.float32), model_recog)
mean_labels = np.mean(np.max(labels, axis=1))
labels_max = labels[np.max(labels, axis=1)>mean_labels]
std_labels_max = np.std(np.max(labels_max, axis=1))
max_labels_max = np.max(np.max(labels_max, axis=1))
labels_max = labels_max[np.where(labels_max>(max_labels_max - std_labels_max))[0]]

prediction = set(np.argmax(labels_max, axis=1))
