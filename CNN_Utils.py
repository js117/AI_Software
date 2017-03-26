# STACK: - Anaconda3 (Python 3.5.2, Anaconda 4.2.0 (64-bit): C:\Program Files\Anaconda3
#		 - TensorFlow backend (see: C:\Users\JDS\.keras\keras.json)
#		 - pip install --upgrade --ignore-installed tensorflow

import h5py
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Reshape, merge
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, SGD
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.layers.core import Lambda, Merge
from keras.engine import Layer

# Choose an output layer, e.g. 'block3_pool'
# For choices, see: https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py 
def GetVGGModel(output_layer_str, W, H, print_timing):
	if print_timing == 1:
		ms1 = time.time()*1000.0
	
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=[W, H, 3])
	model = Model(input=base_model.input, output=base_model.get_layer(output_layer_str).output)
	
	if print_timing == 1:
		ms2 = time.time()*1000.0
		print('VGG model loaded. Time elasped: ' +str(round((ms2-ms1)/1000, 2)) + ' secs')
	return model
	
 # x is image (N x W x H x 3) normalized appropriately, model is VGG	
def VGGFeatures(model, x, print_timing):
	if print_timing == 1:
		ms1 = time.time()*1000.0
		
	features = model.predict(x)
	features = features / np.max(features) # normalize from 0 --> 1 for both our new network inputs, and for visualization
	
	if print_timing == 1:
		ms2 = time.time()*1000.0
		print('VGG predict time: ' +str(round((ms2-ms1), 4)) + ' millis')
		
	return features
	
# x is of shape (n, W, H, 3) and is normalized already
# rot_range: [1,360] / w/h_shift_range: [0,1] / shr_range: [0,1] / h_flip = True,False / v_flip = True,False	
def AugmentImageDataInMem(x, multiplier, rot_range, w_shift_range, h_shift_range, shr_range, zm_range, h_flip, v_flip): 
	n = x.shape[0]
	train_datagen = ImageDataGenerator(
										rotation_range=rot_range,
										width_shift_range=w_shift_range,
										height_shift_range=h_shift_range,
										shear_range=shr_range,
										zoom_range=zm_range,
										horizontal_flip=h_flip,
										vertical_flip=v_flip,
										fill_mode='nearest')
	w, h, c = x.shape[1:]
	auged = np.zeros((round(n*multiplier), w, h, c))
	
	for i in range(n*multiplier):
		auged[i,:] = train_datagen.random_transform(x[randint(0,n-1),:])
		
	return np.concatenate((x, auged), axis=0) # returns original data + aug'd data

############################################ v HEATMAP v ################################################
# https://github.com/jacobgil/keras-grad-cam
	
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes)) # tf.mul ? 

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize_grad_cam(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
	
# image is preprocessed: normalize to 0-1, expand_dims	
def grad_cam_heatmap(input_model, image, CAM_W, CAM_H, nb_classes, category_index, layer_name): 
    model = Sequential()
    model.add(input_model)

    #nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize_grad_cam(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :] # e.g. output is size 14x14 x 20(feature maps)

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (CAM_W, CAM_H))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    #image -= np.min(image)
    #image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)
	
############################################ ^ HEATMAP ^ ################################################	