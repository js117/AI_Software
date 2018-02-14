# STACK: - Anaconda3 (Python 3.5.2, Anaconda 4.2.0 (64-bit): C:\Program Files\Anaconda3
#		 - TensorFlow backend (see: C:\Users\JDS\.keras\keras.json)
#		 - pip install --upgrade --ignore-installed tensorflow

import time
ms1 = 0; ms2 = 0
print("\n\n --- Loading ML libraries (about 10 secs)... --- \n\n\n")
ms1 = time.time()*1000.0

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path
import sys
from sys import platform

ms2 = time.time()*1000.0
print("\nLoaded ML libraries. Time elasped: ", int(ms2-ms1), " ms\n")

MODEL_STRING_NAME = 'my_model.h5'
VIDEO_FRAMES_FOLDER_NAME = 'sequence_video_frames'


'''
if not os.path.exists(MODEL_STRING_NAME): 
	print("\n\n --- Loading initial model weights from net... --- \n\n")
	ms1 = time.time()*1000.0
	# first time: download weights we need from online and save. Else load locally.
	base_model = VGG16(include_top=False,weights='imagenet')
	for layer in base_model.layers:
		layer.trainable = False
	base_model.compile(loss='mean_squared_error', optimizer='sgd') # who cares, not gonna train this.
	
	model = Model(input=base_model.input, output=base_model.get_layer('block5_pool').output) 
	# Use last pool layer to avoid getting whole weight file (~10x less memory).
	# Need to flatten after model.predict, in order to use as input to other models.
	
	model.save(MODEL_STRING_NAME)
	ms2 = time.time()*1000.0
else:
	print("\n\n --- Loading initial model weights locally... --- \n\n")
	ms1 = time.time()*1000.0
	model = load_model(MODEL_STRING_NAME)
	ms2 = time.time()*1000.0

print("\nLoaded VGG16 model. Time elasped: ", int(ms2-ms1), " ms\n")
# From web: about 166s (2 mins 46 secs)
# From local: about 4 secs
# Note: gives warning: 'No training configuration found in save file:' but we don't care (?) b/c we're not training it
'''

##################### BASIC TESTING #######################
# Loading/viewing images, getting flattened VGG features

#img_path1 = 'elephant1.jpg'; img1 = image.load_img(img_path1, target_size=(224, 224))
#img_path2 = 'elephant2.jpg'; img2 = image.load_img(img_path2, target_size=(224, 224))
#img_path3 = 'skyscrapers.jpg'; img3 = image.load_img(img_path3, target_size=(224, 224))
# View an image, resized: 
#cv2.namedWindow("img1", cv2.WINDOW_AUTOSIZE); cv2.imshow("img1", cv2.resize(cv2.imread(img_path1), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img2", cv2.WINDOW_AUTOSIZE); cv2.imshow("img2", cv2.resize(cv2.imread(img_path2), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img3", cv2.WINDOW_AUTOSIZE); cv2.imshow("img3", cv2.resize(cv2.imread(img_path3), (224, 224))); cv2.waitKey(0)

#ms1 = time.time()*1000.0
#x1 = preprocess_input(np.expand_dims(image.img_to_array(img1), axis=0))
#x2 = preprocess_input(np.expand_dims(image.img_to_array(img2), axis=0))
#x3 = preprocess_input(np.expand_dims(image.img_to_array(img3), axis=0))

#features1 = (model.predict(x1)).reshape(-1)
#features2 = (model.predict(x2)).reshape(-1)
#features3 = (model.predict(x3)).reshape(-1)

#print(np.mean(np.abs(np.subtract(features1, features2))))
#print(np.mean(np.abs(np.subtract(features1, features3))))
#print(np.mean(np.abs(np.subtract(features2, features3))))

#print(features1[1234:1234+25]); print(features2[1234:1234+25]); print(features3[1234:1234+25])

#ms2 = time.time()*1000.0
#print("Testing took: ", int(ms2-ms1), " ms")

### TEST PLAYING VIDEO, create frame database if not exist already: 

if not os.path.exists(VIDEO_FRAMES_FOLDER_NAME):
	os.makedirs(VIDEO_FRAMES_FOLDER_NAME)

	cap = cv2.VideoCapture('sample_sequence_downsampled.mp4')
	numFrames = 0
	while True:
		if cap.grab():
			flag, frame = cap.retrieve()
			numFrames = numFrames + 1
			name = "%d.jpg"%numFrames
			cv2.imwrite(VIDEO_FRAMES_FOLDER_NAME+"/"+name, frame)     # save frame as JPEG file
			if not flag:
				continue
			else:
				cv2.imshow('video', frame)
		if cv2.waitKey(10) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

	print("Num frames: ", numFrames) # expected: 15 fps * 40 secs = 600 frames


# First, prepare data similar to mnist:
'''from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data() # _ argument would be y, i.e. the actual #
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape); print(x_test.shape)
#(60000, 28, 28)
#(10000, 28, 28)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))		# WATCH DIM ORDERING FOR TENSORFLOW BACKEND
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
## Compress for testing: 
x_train = x_train[0:4000,:,:,:]
x_test = x_test[0:400,:,:,:]
print(x_train.shape); print(x_test.shape)
#(60000, 1, 28, 28)
#(10000, 1, 28, 28) # WATCH DIM ORDERING FOR TENSORFLOW BACKEND

#sys.exit()

################### DEEP CONVOLUTIONAL AUTOENCODER -- TEST #########################
# test on MNIST
#####################################################################################
input_img = Input(shape=(28, 28, 1))


x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same', name='encoded')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', name='start_decode')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='decoded')(x)

autoencoder = Model(input_img, decoded)

############################ ENCODER / DECODER ############################
# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=autoencoder.get_layer("encoded").output_shape[1:])
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.get_layer("start_decode") #autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
############################################################################
# HOW TO USE:
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
############################################################################

#object = autoencoder.get_layer("encoded")
#print([method for method in dir(object) if callable(getattr(object, method))])
#sys.exit()

# per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tb = TensorBoard(log_dir='logs')
autoencoder.fit(x_train, x_train,
                nb_epoch=40,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tb])
				
ae_str_name = 'ae_test_digits'+str(time.time()).replace(".","")+'.h5'
autoencoder.save(ae_str_name)
				
import matplotlib.pyplot as plt

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

################### DEEP CONVOLUTIONAL AUTOENCODER -- APP #########################
# Idea: given a short video sequence with a stationary setting, find  
# a set of compressed features for each frame. We will then use these
# to train our LSTM. At first, this training will be done separately. 
# Later we can try trying compressed features + sequence prediction jointly.
####################################################################################

# TODO:
# 1) Deep Convolutional Autoencoder, with 1/2 frames for train, 1/2 for test (change ratios?)
# 2) Basic LSTM sequence model
# Note: have visualizations during training (e.g. via TensorBoard)
# Extra: compare DCA features w/ VGG features trained on 2 LSTMs
# 
# Subtasks: 
# - Prepare data similar to MNIST
# - Create DCA model
# - Train w/ TensorBoard visualization
# - google: "keras Sequence classification with LSTM"

# Create training and testing data: approx. 70 - 30 
curr_dir = os.getcwd()
search_dir = ""
if platform == "win32":
	search_dir = os.getcwd()+"\\"+VIDEO_FRAMES_FOLDER_NAME
else:
	search_dir = os.getcwd()+"/"+VIDEO_FRAMES_FOLDER_NAME
os.chdir(search_dir)
files = filter(os.path.isfile, os.listdir(search_dir))
files = [os.path.join(search_dir, f) for f in files] # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))
os.chdir(curr_dir)
# Now our files are sorted chronologically (they were created in this order)
# and we can put into training data, similar to below format:

#(60000, 28, 28, 1) # train (num_examples, width, height, channels)
#(10000, 28, 28, 1) # test (num_examples, width, height, channels)
WIDTH = 112; HEIGHT = 112
NUM_CHANNELS = 3
X_TRAIN = []; Y_TRAIN = []; # Note: for autoencoders, X_TRAIN == Y_TRAIN so we won't create Y_TRAIN
X_TEST = []; Y_TEST = []; # ^ Ditto
itr = 0
ratio_testing = 0.25
print("Preparing video files for autoencoder training...")
for file in files:
	itr = itr + 1
	#print(file); 

	x1 = np.expand_dims(image.img_to_array(image.load_img(file, target_size=(WIDTH, HEIGHT))), axis=0)
	#x1 = preprocess_input(x1)
	x1 = np.reshape(x1, (len(x1), WIDTH, HEIGHT, NUM_CHANNELS))
	
	# Add to test data:
	if itr % int(1/ratio_testing) == 0:
		X_TEST.append(x1[0,:])
	# Add to training data:
	else:
		X_TRAIN.append(x1[0,:])
		
X_TRAIN = np.asarray(X_TRAIN); Y_TRAIN = np.asarray(Y_TRAIN);
X_TEST = np.asarray(X_TEST); Y_TEST = np.asarray(Y_TEST);
X_TRAIN = X_TRAIN.astype('float32') / 255.
X_TEST = X_TEST.astype('float32') / 255.
print("Video files prepared. Training and testing dataset shapes below: ")		
print(X_TRAIN.shape)
print(X_TEST.shape)
num_frames_train = X_TRAIN.shape[0]
num_frames_test = X_TEST.shape[0]

############################## DEEP AUTOENCODER #####################################
input_img = Input(shape=(WIDTH, HEIGHT, NUM_CHANNELS)); print("Input shape:",(WIDTH, HEIGHT, NUM_CHANNELS))
# Model
#x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='encoded_temp')(x)
encoded = MaxPooling2D((2, 2), border_mode='same', name='encoded')(x)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='start_decode')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
decoded = Convolution2D(NUM_CHANNELS, 3, 3, activation='relu', border_mode='same', name='decoded')(x)

autoencoder = Model(input_img, decoded)

print("--------------------------")
for layer in autoencoder.layers:
	print(layer.output_shape)

############################ ENCODER / DECODER ############################
# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded input
encoded_input_shape = autoencoder.get_layer("encoded").output_shape[1:] 
print("Encoded input shape: ",encoded_input_shape)
encoded_input = Input(shape=encoded_input_shape)
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.get_layer("start_decode") #autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
############################################################################
# HOW TO USE:
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
############################################################################

#object = autoencoder.get_layer("encoded")
#print([method for method in dir(object) if callable(getattr(object, method))])
#sys.exit()

# per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tb = TensorBoard(log_dir='logs')
# Y targets are the video frames that are +1 frame ahead of their respective X input frames
autoencoder.fit(X_TRAIN[1:num_frames_train-1], X_TRAIN[2:num_frames_train],
                nb_epoch=500,
                batch_size=36,
                shuffle=True,
                validation_data=(X_TEST[1:num_frames_test-1], X_TEST[2:num_frames_test]),
                callbacks=[tb])
# 40 epochs on batch_size 100 took [80 minutes]
				
ae_str_name = 'ae_test'+str(time.time()).replace(".","")+'.h5' # pro-tip: manually re-name after each run... 
autoencoder.save(ae_str_name)
				
print("Finished training.")
