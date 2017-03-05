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
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path
import sys

ms2 = time.time()*1000.0
print("\nLoaded ML libraries. Time elasped: ", int(ms2-ms1), " ms\n")

MODEL_STRING_NAME = sys.argv[1]; print(sys.argv[1])
VIDEO_FRAMES_FOLDER_NAME = 'sequence_video_frames' # precondition: should already exist

############### LOAD THE MODEL FROM CMD LINE ARGUMENT (.h5 file) ###############
base_model = load_model(MODEL_STRING_NAME) #VGG19(weights='imagenet')
encoder_model = Model(input=base_model.input, output=base_model.get_layer('encoded').output)
#decoder_model = Model(input=base_model.get_layer('start_decode').input, output=base_model.get_layer('decoded').output)
################################################################################

curr_dir = os.getcwd()
search_dir = os.getcwd()+"\\"+VIDEO_FRAMES_FOLDER_NAME
os.chdir(search_dir)
files = filter(os.path.isfile, os.listdir(search_dir))
files = [os.path.join(search_dir, f) for f in files] # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))
os.chdir(curr_dir)	

########################### From here below: must be same params as vnpf_source.py ###########################
WIDTH = 112; HEIGHT = 112
NUM_CHANNELS = 3
X_TRAIN = []; Y_TRAIN = []; # Note: for autoencoders, X_TRAIN == Y_TRAIN so we won't create Y_TRAIN
X_TEST = []; Y_TEST = []; # ^ Ditto
X_ALL = [];
itr = 0
ratio_testing = 0.25
print("Preparing video files for model testing...")
for file in files:
	itr = itr + 1
	#print(file); 

	x1 = np.expand_dims(image.img_to_array(image.load_img(file, target_size=(WIDTH, HEIGHT))), axis=0)
	#x1 = preprocess_input(x1)
	x1 = np.reshape(x1, (len(x1), WIDTH, HEIGHT, NUM_CHANNELS))
		
	X_ALL.append(x1[0,:])
	
X_ALL = np.asarray(X_ALL)
X_ALL = X_ALL.astype('float32') / 255.
	
X_ALL = np.asarray(X_ALL)
#x_train = x_train.astype('float32') / 255. # Do we need div 255? 
#x_test = x_test.astype('float32') / 255.
print("Video files prepared. Shapes below: ")		
print(X_ALL.shape)
###############################################################################################################

#decoded_imgs = autoencoder.predict(X_TEST) # below should be equivalent
#print("Predicting video sequence, please wait..."); ms1 = time.time()*1000.0
#encoded_imgs = encoder_model.predict(X_ALL); 
#decoded_imgs = base_model.predict(X_ALL); 
#print("Time elasped (s): ",str((time.time()*1000.0 - ms1)/1000))

print("X_ALL shape: ",X_ALL.shape)
#print("Encoded images shape: ",encoded_imgs.shape)
#print("Decoded images shape: ",decoded_imgs.shape)

num_frames = X_ALL.shape[0]
prev_frame = np.array([X_ALL[0,:,:,:]]) # to make shapes work initially.. 
RECREATED_FRAMES = np.zeros(X_ALL.shape)

print("\n Preparing sample output video...\n")
NUM_FRAMES_RESET = 5
for i in range(1, num_frames): 
	#frame = base_model.predict(np.array([X_ALL[i,:,:,:]])) # frame is of size (1, num_channels, width, height)
	# ^ Do the above for frame-by-frame recreation; 
	# This is fairly trivial and gets good as we are just overfitting visual features to video file
	
	# Below: the fun part - can we seed only the 1st frame of the sequence and recover the rest of it? 
	# What we expect: degrading quality video that mostly gets it right (yep)
	frame = base_model.predict(prev_frame)
	RECREATED_FRAMES[i] = frame	
	
	# ADD A RESET: since we degrade quite a bit, let's reset by using an actual frame
	# Mathematically: using x(t+1) ~ p( x(t+1) | x(t) ) will degrade over time, hence the reset 
	if i%NUM_FRAMES_RESET == 0:
		prev_frame = np.array([X_ALL[i,:,:,:]])
	else:
		prev_frame = frame  
	
	
print(RECREATED_FRAMES.shape)

# Get ready to create output videos: 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (2*WIDTH,HEIGHT))
	
print("Creating comparison video: ")
for i in range(0, num_frames):
	img1 = X_ALL[i,:,:,:]
	img2 = RECREATED_FRAMES[i,:,:,:]
	vis = np.concatenate((img1, img2), axis=1)
	cv2.imshow('video', vis)
	cv2.waitKey(100) # play at ~10 fps
	out.write((vis * 255.0).astype('u1')) # don't forget actual videos can't be b/w 0 and 1
	
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

'''
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
'''