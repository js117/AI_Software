# Using Anaconda 3.5
# 
# V-REP instructions: https://www.youtube.com/watch?v=SQont-mTnfM 
# http://developers-club.com/posts/268313/ 

import time
ms1 = 0; ms2 = 0
print("\n\n --- Loading ML libraries (about 10 secs)... --- \n\n\n")
ms1 = time.time()*1000.0

import h5py
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Reshape, merge
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.layers import Lambda
#from keras.utils.visualize_util import plot 
# Download + install "graphviz-2.38.msi" from http://www.graphviz.org/Download_windows.php
# conda install graphviz
# pip install git+https://github.com/nlhepler/pydot.git 


ms2 = time.time()*1000.0
print("\nLoaded ML libraries. Time elasped: ", int(ms2-ms1), " ms\n")

from keras.constraints import Constraint
from keras import backend as K

import numpy as np
import cv2
import os.path
import sys
import msvcrt # WINDOWS ONLY (what does Linux need?)
from sys import platform
from random import randint
from math import sqrt

############################################## GLOBAL VARIABLES #########################################
global CAM_W
global CAM_H
global CAM_C
global NUM_CAM_CHANNELS
global NUM_CAMS
global ACTION_LEN
global vgg_weights_path

CAM_W = 128
CAM_H = 128
NUM_CAM_CHANNELS = 3
NUM_CAMS = 4
CAM_C = NUM_CAM_CHANNELS*NUM_CAMS 		#R,G,B x 4 cameras
ACTION_LEN = 7
ACTION_FEATURE_LEN = ACTION_LEN*3 # e.g. pass in theta, cos(theta), sin(theta)
vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' # NOTE: assumes we use TF backend

def GetFoldersForRuns():
	folders = []
	for x in os.listdir(os.getcwd()):
		if x.startswith('u_sequence'):
			folders.append(x)
	
	return folders
	
def LoadFramesActionsFromFolder(folder):

	curr_dir = os.getcwd()
	search_dir = ""
	if platform == "win32":
		search_dir = os.getcwd()+"\\"+folder # WINDOWS
	else:
		search_dir = os.getcwd()+"/"+folder # LINUX
	os.chdir(search_dir)
	files = filter(os.path.isfile, os.listdir(search_dir))
	files = [os.path.join(search_dir, f) for f in files] # add path to each file
	files.sort(key=lambda x: x) #os.path.getmtime(x))
	os.chdir(curr_dir)
	
	timesteps = int(len(files) / 2)
	#print(len(files)); print(timesteps)
	frames = np.zeros((timesteps, CAM_W, CAM_H, CAM_C), dtype=np.uint8)	# IMPORTANT!! Or else value copy fails.. 
	actions = np.zeros((timesteps, ACTION_LEN)) # default float type should be fine
	i1 = 0
	i2 = 0
	
	for f in files:
		if f.endswith('_x.npy'):		# frame
			#print(f)
			temp = np.load(f)
			frames[i1,:,:,:] = temp[:,:,:]
			i1 = i1 + 1
		elif f.endswith('_u.npy'):		# action
			#print(f)
			actions[i2,:] = np.load(f).reshape(ACTION_LEN)
			i2 = i2 + 1
		else:
			print("ERROR: found file in %s not ending with _x.npy or _u.npy"%folder)
			print("File: %s"%f)
			return
			
	frames = (frames.astype(float) - 127.5) / 255 # normalize
	
	return frames, actions

def PreprocessImgFromDatastore(img): # img is W x H x 3
	# Call this function before displaying, and also before inputting to our network.
	# NOTE: displaying will swap RGB --> BGR, so don't worry if colors look off vs. original
	# PRE-PROCESS:
	# 1. Convert to float. ---> done already
	# 2. Subtract imagenet means
	# 3. Divide by 255
	image = img.astype(float)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	image = image / 255
	#image = image[0,:]
	#print(image.shape)
	return image # 1 x W x H x 3 ---> ready to go into VGG
	
	
def GetFrame4(frame, isVertical): 
### NOTE: imshow does weird things if input is float:
# "If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255]."

	img1 = frame[:,:,0:3]
	img2 = frame[:,:,3:6]
	img3 = frame[:,:,6:9]
	img4 = frame[:,:,9:12]
	if (isVertical == 1):
		img = np.concatenate((np.concatenate((img1, img2),axis=0), np.concatenate((img3, img4),axis=0)), axis=0)
	else:
		img = np.concatenate((np.concatenate((img1, img2),axis=1), np.concatenate((img3, img4),axis=1)), axis=0)
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	return img
	
def ViewFutureFrames(frames): # assumes frames of shape (NFF, CAM_W, CAM_H, CAM_C)
	num_frames = frames.shape[0] #
	
	font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
	
	imgs = GetFrame4(frames[0,:], 1)
	#cv2.putText(imgs,'OpenCV',(5,20), font, 1,(255,255,255),1,cv2.LINE_AA)
	for t in range(1, num_frames):
		next_img = GetFrame4(frames[t,:], 1)
		#cv2.putText(next_img,'[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]',(5,10), font, 0.2,(20,255,20),1,cv2.LINE_AA)
		#cv2.putText(next_img,'[0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71]',(5,25), font, 0.2,(255,20,20),1,cv2.LINE_AA)
		imgs = np.concatenate((imgs, next_img), axis=1)
		
	return imgs
	
	


def GetVGGModel():
	ms1 = time.time()*1000.0
	
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=[CAM_W, CAM_H, 3])
	# Pick your output activation. See source:
	# https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py 
	model = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output) # should be something like 32x32x128, since we started w/ 128x128 img
	
	ms2 = time.time()*1000.0
	print('VGG model loaded. Time elasped: ' +str(round((ms2-ms1)/1000, 2)) + ' secs')
	return model
	
def VGGFeatures(model, x): # x is image (W x H x 3), model is VGG
	features = model.predict(x)
	features = features / np.max(features) # normalize from 0 --> 1 for both our new network inputs, and for visualization
	return features
	
def FramesToVGGFeatures(VGG_model, frames): # input: 128x128x12 frames for 1 timestep of the 4 cams
	vgg_cam_1 = VGGFeatures(VGG_model, frames[:,:,0:3])
	vgg_cam_1 = VGGFeatures(VGG_model, frames[:,:,3:6])
	vgg_cam_1 = VGGFeatures(VGG_model, frames[:,:,6:9])
	vgg_cam_1 = VGGFeatures(VGG_model, frames[:,:,9:12])
	
	return np.concatenate((vgg_cam_1,vgg_cam_2,vgg_cam_3,vgg_cam_4), axis=2) # 32 x 32 x 512
	
	
def VisualizeVGGFeatures(f, x): 
	# f is of size (W/4, H/4, 128) or in our case, 32 x 32 x 128 (coincidence that N_feature_map = 128). 
	# input x is the frame we put in, has size (W, H, 3) or 128 x 128 x 3
	f_sizes = f.shape[0]
	return 0
	
##################################################### MODELS ##########################################################
# Generator, Discriminator Architecture
#
# IDEA: generator produces x(t+1), u(t) from x(t), jointly learns dynamics and control by predicting the future evolution
#		of both, based on the data of what the expert does. 
# 
# 		G : p(x(t+1), u(t) | x(t))
#
# then we train by chaining the next states together: 
#
# x(t+1), u(t)   = G(x(t)) 			---> only x(t) is ground truth input fed into network
# x(t+2), u(t+1) = G(x(t+1))  		---> now we're using simulated next state predictions to plan future actions
# x(t+3), u(t+2) = G(x(t+2))  		---> 2nd step simulated prediction
# ...
# x(t+F), u(t+F-1) = G(x(t+F-1))  	---> F steps predicted
#
# To evaluate the predictive quality, we can try to fool the discriminator with {x(tau), u(tau)}[tau = t,...,t+F-1] 
# when the real data comes from the expert demonstrations 
#
# DISCRIMINATOR:
#
# (frame, action) ---> D ---> scalar output (non-sigmoid) multiplied by NUM_FUTURE_FRAMES. 
# We feed in either expert (frame, action) pairs for ground truth, or generator generated (f, a) pairs
#
# OPTIMIZER:
#
# Following the WGANs approach, https://arxiv.org/pdf/1701.07875.pdf 
# --> scalar (non-sigmoid) discriminator output, weight clipping, RMSProp, lr = 0.00005, 
#	  20 - 60 timesteps/samples, 5 disc. itrs/gen. itr (with initial disc. training for a bunch of epochs, at least until some saturation)
#
	
# Model Variables
global CLIP
global NUM_VGG_FEAT_MAPS
global NUM_FUTURE_FRAMES
global NUM_PAST_FRAMES
global VGG_FEAT_W
global VGG_FEAT_H 
global NOISE_DIM
global NUM_NOISE_CHANNELS
global CONV1_FEAT_MAPS
global CONV2_FEAT_MAPS
global CONV3_FEAT_MAPS
global CONVP1_FEAT_MAPS 
global CONVP2_FEAT_MAPS
global CONVF1_FEAT_MAPS
global CONVF2_FEAT_MAPS
global CONVD1_FEAT_MAPS
global CONVD2_FEAT_MAPS
global CONVD3_FEAT_MAPS
global D_DENSE1
global D_DENSE2
global D_DENSE3
global AP_BRANCH_DENSE1
global AP_BRANCH_DENSE2
CLIP = 0.1
VGG_FEAT_W = 32
VGG_FEAT_H = 32
NUM_VGG_FEAT_MAPS = 128 # per camera
NUM_FUTURE_FRAMES = 3
NUM_PAST_FRAMES = 3
NOISE_DIM = ACTION_LEN*NUM_FUTURE_FRAMES
NUM_NOISE_CHANNELS = 1
CONV1_FEAT_MAPS = 200
CONV2_FEAT_MAPS = 100
CONV3_FEAT_MAPS = 50
CONV4_FEAT_MAPS = 12
CONVP1_FEAT_MAPS = round(CAM_C*NUM_PAST_FRAMES/2) # use an even number of past frames please; e.g. 12*15 = 90
CONVP2_FEAT_MAPS = round(CAM_C*NUM_PAST_FRAMES/4) # use an even number of past frames please; e.g. 45
CONVP3_FEAT_MAPS = NUM_PAST_FRAMES 				  # saving 1 (CAM_W x CAM_H) map per past frame; e.g. 15 
AP_BRANCH_DENSE1 = 400 
AP_BRANCH_DENSE2 = 100
CONVF1_FEAT_MAPS = round(CAM_C*NUM_FUTURE_FRAMES/2) # e.g. if future_frames is 15, this is 90
CONVF2_FEAT_MAPS = round(CAM_C*NUM_FUTURE_FRAMES/4) # ^ is 45 now
CONVF3_FEAT_MAPS = NUM_FUTURE_FRAMES				# e.g. 15
CONVD1_FEAT_MAPS = round(CAM_C*NUM_FUTURE_FRAMES) 	# e.g. 180
CONVD2_FEAT_MAPS = round(CONVD1_FEAT_MAPS/2)		# e.g. 90
CONVD3_FEAT_MAPS = round(CONVD2_FEAT_MAPS/2)		# e.g. 45
CONVD4_FEAT_MAPS = round(CONVD3_FEAT_MAPS/2)		# e.g. 22
CONVD5_FEAT_MAPS = round(CONVD4_FEAT_MAPS/3)		# e.g. 7
D_DENSE1 = 2500										# otherside going into this would be e.g. 32x32x8 = 8192
D_DENSE2 = 1000
D_DENSE3 = 250
D_DENSE_OUT = 1

def GeneratorModel():
	#	INPUT: combine 4 main sources
	#
	#	1) frame at time t, f(t) (where "frame" means the 4-tuple of cam images, i.e. 128x128x12).
	#	   ---> size: 128 x 128 x 4
	#	
	#	2) previous frames: [f(t-1), f(t-2), ..., f(t-p-1)]
	#	   We will concat p previous frames together, have learnable conv+pool layers to choose what to "remember" from past 
	#	   (if prev timestep not available: zeros)
	#	   ---> size: 128 x 128 x 12p=180 (e.g. p = 15) conv'd down to 128 x 128 x 15
	#
	#	3) previous actions: [a(t-1), a(t-2), ..., a(t-p-1)]
	#	   We will concat p previous actions together, have learnable projection, reshape to merge with other inputs. 
	#	   Similar idea: learn what to remember from the past. (if prev timestep not available: zeros)
	#	   ---> size: 7 x p (e.g. p = 10)
	#	   ---> reshape from e.g. 70 --> 16384 (128*128*1)
	#
	#	4) noise: N(0,1)
	#	   ---> size: start w/ 49-len vector, project and reshape from 49 --> 16384 (= 128*128*1)
	#
	#	== Total size of merged input:
	#		128 x 128 x 21
	#
	#	OUTPUT:
	#	
	#	1) Next frames prediction: 			{f_(t+1)} for # of future frames
	#	2) Action prediction to take at t: 	{u_(t)}	  for # of future frames
	
	# INPUT#1: frame(t): 
	input_curr_frame = Input(shape=(CAM_W, CAM_H, CAM_C), name='input_curr_frame') # 128 x 128 x 12
	
	# INPUT#2: prev_frames
	input_prev_frames_raw = Input(shape=(CAM_W, CAM_H, CAM_C*NUM_PAST_FRAMES), name='input_prev_frames_raw')
	# Learn what's important to remember from the past: 
	input_prev_frames = Convolution2D(CONVP1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames_raw)
	input_prev_frames = Convolution2D(CONVP2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames)
	input_prev_frames = Convolution2D(CONVP3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames) # will now be 128 x 128 x CAM_C=12
	
	# INPUT#3: prev_actions
	input_prev_actions_raw = Input(shape=(ACTION_LEN*NUM_PAST_FRAMES,), name='input_prev_actions_raw')
	input_prev_actions_p = Dense(output_dim=CAM_W*CAM_H*1, activation='elu')(input_prev_actions_raw)
	input_prev_actions_p_r = Reshape((CAM_W, CAM_H, 1), input_shape=(CAM_W*CAM_H*1,), name='input_prev_actions_p_r')(input_prev_actions_p)
	
	# INPUT#4: noise
	input_noise = Input(shape=(NOISE_DIM,), name='input_noise') # now project and reshape noise:
	noise_p = Dense(output_dim=CAM_W*CAM_H*NUM_NOISE_CHANNELS, activation='elu')(input_noise)
	noise_p_r = Reshape((CAM_W, CAM_H, NUM_NOISE_CHANNELS), input_shape=(CAM_W*CAM_H*NUM_NOISE_CHANNELS,), name='noise_p_r')(noise_p)
	
	# MERGE INPUTS: 
	merged_input = merge([input_curr_frame, input_prev_frames, input_prev_actions_p_r, noise_p_r], mode='concat', concat_axis=3, name='merged_input___0') 
	# should be e.g. (None, 128, 128, ~21)
	
	# Pre-process merged inputs: 
	merged_input = Convolution2D(CAM_C, 3, 3, activation='elu', border_mode='same', name='merged_input___1')(merged_input) # CAM_C+1 is size in loop
	merged_input = Convolution2D(CAM_C, 3, 3, activation='elu', border_mode='same', name='merged_input___2')(merged_input)
	merged_input = Convolution2D(CAM_C, 3, 3, activation='elu', border_mode='same', name='merged_input___3')(merged_input)
	
	# Predicting the future: "mix and demix" architecture.
	# The way we're doing it now, frame(t+1) and action(t) get "special treatment" as they take advantage of VGG features. 
	# next_actions = [] 										# add: a_p(t)   --> next predicted action (causes next predicted frame)
	# next_frames = []										# add: f_p(t+1) --> next predicted frame

	# TODO 
	
	SHARED_C1 = Convolution2D(CONV1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='SHARED_C1')
	SHARED_C2 = Convolution2D(CONV2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='SHARED_C2')
	SHARED_C3 = Convolution2D(CONV3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='SHARED_C3')
	SHARED_C4 = Convolution2D(CONV4_FEAT_MAPS, 3, 3, activation='tanh', border_mode='same', name='SHARED_C4')
	
	for t in range(0, NUM_FUTURE_FRAMES):
	
		if (t == 1):
			next_frames = nfp_branch
			next_actions = action_prediction
			
		# OUTPUT First branch: predict action to take at this frame (ap_branch)
		ap_branch = Convolution2D(CONV1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(merged_input)
		ap_branch = MaxPooling2D((2, 2), border_mode='same')(ap_branch)	# out = e.g. 64 x 64 x CONV1_FEAT_MAPS
		ap_branch = Convolution2D(CONV2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(ap_branch)
		ap_branch = MaxPooling2D((2, 2), border_mode='same')(ap_branch)	# out = e.g. 32 x 32 x CONV2_FEAT_MAPS 
		ap_branch = Convolution2D(CONV3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(ap_branch)
		ap_branch = MaxPooling2D((2, 2), border_mode='same')(ap_branch)	# out = e.g. 16 x 16 x CONV3_FEAT_MAPS
		ap_branch = Convolution2D(CONV4_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(ap_branch)
		ap_branch = Flatten()(ap_branch) # e.g. output should be e.g. 16*16*12 = 3072
		ap_branch = Dense(output_dim=AP_BRANCH_DENSE1, activation='elu')(ap_branch)
		ap_branch = Dense(output_dim=AP_BRANCH_DENSE2, activation='elu')(ap_branch)
		action_prediction = Dense(output_dim=ACTION_LEN, activation='tanh', name='n_a___'+str(t))(ap_branch)
		# ADD TO OUTPUTS:
		#next_actions.append(action_prediction)
		
		# Create a connection from pre-output of action_prediction and input to conv layers of the next frame prediction branch: 
		ap_branch_p = Dense(output_dim=CAM_W*CAM_H*1, activation='elu')(ap_branch) 
		ap_branch_p_r = Reshape((CAM_W, CAM_H, 1), input_shape=(CAM_W*CAM_H*1,), name='ap_branch_p_r___'+str(t))(ap_branch_p)

		# OUTPUT Second branch: next frame prediction (nfp_branch)
		# Idea with shared layers: ACTION choice can vary in the future, but how we transform to new observation (dynamics model) is constant all the time
		nfp_branch = merge([merged_input, ap_branch_p_r], mode='concat', concat_axis=3, name='merged_input_nfp_branch___'+str(t)) 
		nfp_branch = SHARED_C1(nfp_branch)
		nfp_branch = SHARED_C2(nfp_branch)
		nfp_branch = SHARED_C3(nfp_branch)
		nfp_branch = SHARED_C4(nfp_branch) # now output is same dim as frames
		# ADD TO OUTPUTS:
		#next_frames.append(nfp_branch)
		
		if (t >= 1):
			next_frames = merge([next_frames, nfp_branch], mode='concat', concat_axis=3)
			next_actions = merge([next_actions, action_prediction], mode='concat', concat_axis=1)
		
		# Create mixed input for next iteration: 
		if (t < NUM_FUTURE_FRAMES-1):
			merged_input = nfp_branch
			#merged_input = merge([nfp_branch, ap_branch_p_r], mode='concat', concat_axis=3, name='merged_input___'+str(t+1)) 
			
	'''frame = next_frames[0];    #frame.set_shape([1, CAM_W, CAM_H, CAM_C])
	action = next_actions[0];  #frame.set_shape([1, CAM_W, CAM_H, CAM_C])
	future_frames = frame
	future_actions = action
	for t in range(1, NUM_FUTURE_FRAMES):
		frame = next_frames[t]
		future_frames = merge([future_frames, frame], mode='concat', concat_axis=0)
		action = next_actions[t];
		future_actions = merge([future_actions, action], mode='concat', concat_axis=0)'''
	
	# DEFINE MULTI-INPUT, MULTI-OUTPUT MODEL: 
	model_inputs = [input_curr_frame, input_prev_frames_raw, input_prev_actions_raw, input_noise]
	model_outputs = [next_frames, next_actions] 
	model = Model(input=model_inputs, output=model_outputs)
	
	'''RMSpropOpt = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0) # edit me
	model.compile(optimizer=RMSpropOpt,
              loss={'nfp_branch_output': 'binary_crossentropy', 
					'action_prediction_output': 'binary_crossentropy'},
              loss_weights={'nfp_branch_output': 0.5, 
							'action_prediction_output': 0.5})
	'''
	
	return model
	
def DynamicsLoss(y_true, y_pred):
	# model_outputs = [next_frame, input_prev_frame]
	pred = y_pred[0]
	input = y_pred[1]
	mask = K.abs(pred - input) # we care more about pixels that change during the frame
	mask = mask + 0.1*K.mean(mask)
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)	
	
def DynamicsModel():

	# Params
	C1_FEAT_MAPS = CAM_C+1
	C2_FEAT_MAPS = CAM_C+1
	C3_FEAT_MAPS = CAM_C+1
	C4_FEAT_MAPS = CAM_C

	input_prev_frame  = Input(shape=(CAM_W, CAM_H, CAM_C), name='input_curr_frame') # 128 x 128 x 12
	input_prev_action = Input(shape=(ACTION_FEATURE_LEN,), name='input_prev_action')

	input_prev_action_p = Dense(output_dim=CAM_W*CAM_H*1, activation='elu')(input_prev_actions_raw)
	input_prev_action_p_r = Reshape((CAM_W, CAM_H, 1), input_shape=(CAM_W*CAM_H*1,), name='input_prev_actions_p_r')(input_prev_action_p)
	
	merged_input = merge([input_prev_frame, input_prev_action_p_r], mode='concat', concat_axis=3, name='merged_input___0') 
	
	C1 = Convolution2D(C1_FEAT_MAPS, 7, 7, activation='elu', border_mode='same', name='C1')(merged_input)
	C2 = Convolution2D(C2_FEAT_MAPS, 5, 5, activation='elu', border_mode='same', name='C2')(C1)
	C3 = Convolution2D(C3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='C3')(C2)
	C4 = Convolution2D(C4_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='C4')(C3)
	next_frame = Convolution2D(CAM_C, 3, 3, activation='elu', border_mode='same', name='next_frame')(C4)
	
	model_inputs = [input_prev_frame, input_prev_action]
	model_outputs = [next_frame, input_prev_frame]
	model = Model(input=model_inputs, output=model_outputs)

def CriticModel():

	# The discriminator takes a large input - the sequence of future frames and actions predicted for the future - and outputs a scalar value. 
	# This scalar value is trained to be higher on real data, lower on fake data. 
	# 
	# Real input: just from the expert demonstration files. Concat the necessary frames and actions. 
	# Fake input: after the generator predicts its future frames and actions, create the inputs below and pass them in. 
	#
	# Given this sequence of NUM_FUTURE_FRAMES, was this sequence created by an expert or our generator? That is the adversarial game. 
	
	# INPUT#1: future_frames
	
	input_future_frames_raw = Input(shape=(CAM_W, CAM_H, CAM_C*NUM_FUTURE_FRAMES), name='input_future_frames_raw')
	#input_future_frames = Reshape((CAM_W, CAM_H, CAM_C*NUM_FUTURE_FRAMES), input_shape=(NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))(input_future_frames_raw)
	input_future_frames = Convolution2D(CONVF1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_future_frames_raw)
	input_future_frames = Convolution2D(CONVF2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_future_frames)
	input_future_frames = Convolution2D(CONVF3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_future_frames)
	
	# INPUT#2: future_actions
	input_future_actions_raw = Input(shape=(NUM_FUTURE_FRAMES*ACTION_LEN,), name='input_future_actions_raw')
	#input_future_actions_r = Reshape((ACTION_LEN*NUM_FUTURE_FRAMES,), input_shape=(NUM_FUTURE_FRAMES, ACTION_LEN))(input_future_actions_raw)
	input_future_actions_p = Dense(output_dim=CAM_W*CAM_H*1, activation='elu')(input_future_actions_raw)
	input_future_actions_p_r = Reshape((CAM_W, CAM_H, 1), input_shape=(CAM_W*CAM_H*1,), name='input_future_actions_p_r')(input_future_actions_p)
	
	# MERGE INPUTS: 
	merged_input = merge([input_future_frames, input_future_actions_p_r], mode='concat', concat_axis=3, name='merged_input') 
	# should be something like 128,128,181 for 15 future frames
	
	# Downsize and Discriminate: 
	Disc = Convolution2D(CONVD1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(merged_input)
	Disc = MaxPooling2D((2, 2), border_mode='same')(Disc) # e.g. 64 x 64 x 180
	Disc = Convolution2D(CONVD2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(Disc)	
	Disc = Convolution2D(CONVD3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(Disc)
	Disc = MaxPooling2D((2, 2), border_mode='same')(Disc) # e.g. 32 x 32 x 45
	Disc = Convolution2D(CONVD4_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(Disc)
	Disc = Convolution2D(CONVD5_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(Disc)	
	Disc = Flatten()(Disc) # e.g. 32 x 32 x 7 (7168 #s)
	Disc = Dense(output_dim=D_DENSE1, activation='elu')(Disc)
	Disc = Dense(output_dim=D_DENSE2, activation='elu')(Disc)
	Disc = Dense(output_dim=D_DENSE3, activation='elu')(Disc)
	Disc = Dense(output_dim=D_DENSE_OUT, activation='linear')(Disc)	
	
	# DEFINE INPUT, OUTPUT FOR MODEL: 
	model_inputs = [input_future_frames_raw, input_future_actions_raw]
	model_outputs = Disc
	model = Model(input=model_inputs, output=model_outputs)
	
	#RMSpropOpt = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0) # edit me
	#model.compile(optimizer=RMSpropOpt, loss='mean_absolute_error')
	
	return model
	
def DISC_ON_GEN(GEN, DISC):

	# How this model works:
	# Creates a model where the input is the output of the generator, and the overall out is the output of the discriminator
	# Step 1: get generator inputs as input to this model
	# Step 2: 

	# INPUT#1: frame(t): 
	input_curr_frame = Input(shape=(CAM_W, CAM_H, CAM_C), name='input_curr_frame') # 128 x 128 x 12
	# INPUT#2: prev_frames
	input_prev_frames_raw = Input(shape=(CAM_W, CAM_H, CAM_C*NUM_PAST_FRAMES), name='input_prev_frames_raw')
	# INPUT#3: prev_actions
	input_prev_actions_raw = Input(shape=(ACTION_LEN*NUM_PAST_FRAMES,), name='input_prev_actions_raw')
	# INPUT#4: noise
	input_noise = Input(shape=(NOISE_DIM,), name='input_noise') 
	
	gen_inputs = [input_curr_frame, input_prev_frames_raw, input_prev_actions_raw, input_noise]
	gen_outputs = GEN(gen_inputs)
	
	
	print(GEN.output_shape)
	print(gen_outputs[0].get_shape())
	print(gen_outputs[1].get_shape())
	
	#timestamp = str((time.time()*1000))
	#plot(GEN, show_shapes=True, show_layer_names=True, to_file='GEN_'+timestamp+'.png')
	#plot(DISC, show_shapes=True, show_layer_names=True, to_file='DISC_'+timestamp+'.png')

	
	disc_inputs = [gen_outputs[0], gen_outputs[1]]
	disc_output = DISC(disc_inputs)
	
	D_ON_G_MODEL = Model(input=gen_inputs, output=disc_output, name="D_ON_G_MODEL")
	
	return D_ON_G_MODEL
	
	
def ParseGenOutputs(outs):
	future_frames_out = np.zeros((NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))
	future_actions_out = np.zeros((NUM_FUTURE_FRAMES, ACTION_LEN))
	# outs order is NFF of frames, then NFF of actions
	for i in range(0, NUM_FUTURE_FRAMES):
		future_frames_out[i,:] = outs[i][0,:]
		future_actions_out[i,:] = outs[i+NUM_FUTURE_FRAMES][0,:]
		
	return future_frames_out, future_actions_out

def WGAN_LOSS_DISC(y_true, y_pred): # let "y_true" be the DISC prediction for the GENERATED inputs
	return -1*K.mean(y_pred - y_true)	 # linear loss on discriminator outputs
	
def WGAN_LOSS_GEN(y_true, y_pred): # let y_true be 0, doesn't matter
	return -1*K.mean(y_pred - 0)	 # linear loss on discriminator outputs

	
global TOTAL_GANS_TRAINING_ITRS
global SAVE_CHECKPOINT_ITRS
global DISC_PER_GEN_ITRS
global INITIAL_DISC_ITRS
global NUM_EXPERT_DEMONSTRATIONS
global CURR_EXPERT_DEMONSTRATION
global DEMONSTRATION_FOLDERS
global LENGTH_CURR_DEMONSTRATION # e.g. current demo we're looking at is 230 timesteps
global T_CURR_DEMONSTRATION 	 # and e.g. we're currently on timestep 87
global G_PREV_FRAMES_BUFFER		 # Note that G and D will use these buffers slightly differently:
global G_PREV_ACTION_BUFFER		 # G will shift them forward using its own projections of the future, D will use purely demonstration data
global D_PREV_FRAMES_BUFFER		 
global D_PREV_ACTION_BUFFER		 
global DISC_LOSS_BUFFER
global DISC_LOSS_MEAN_MAX

TOTAL_GANS_TRAINING_ITRS = 100000	
SAVE_CHECKPOINT_ITRS = 100
DISC_PER_GEN_ITRS = 10
INITIAL_DISC_ITRS = 20
CURR_EXPERT_DEMONSTRATION = -1 # start on the first one, loop to another one each itr
LENGTH_CURR_DEMONSTRATION = -1
DISC_LOSS_BUFFER = np.zeros((10,1))
GEN_LOSS_BUFFER = np.zeros((10,1))
DISC_LOSS_MEAN_MAX = np.mean(DISC_LOSS_BUFFER)

# For GENERATOR input
PREV_FRAMES_BUFFER = np.zeros((NUM_PAST_FRAMES, CAM_W, CAM_H, CAM_C))
PREV_ACTION_BUFFER = np.zeros((NUM_PAST_FRAMES, ACTION_LEN))
# For DISCRIMINATOR input
FUTURE_FRAMES_BUFFER = np.zeros((NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))
FUTURE_ACTION_BUFFER = np.zeros((NUM_FUTURE_FRAMES, ACTION_LEN))

	
if __name__ == '__main__':
	print("Running program...")
	
	DEMONSTRATION_FOLDERS = GetFoldersForRuns()
	NUM_EXPERT_DEMONSTRATIONS = len(DEMONSTRATION_FOLDERS)
	
	print([f for f in DEMONSTRATION_FOLDERS])
	
	#VGG_model = GetVGGModel()
	
	#frames, actions = LoadFramesActionsFromFolder(DEMONSTRATION_FOLDERS[CURR_EXPERT_DEMONSTRATION])
	
	#print(frames.shape)
	#print(actions.shape)
	
	#imgs = ViewFutureFrames(frames[60:70,:])
	#cv2.imshow('image',imgs) 
	#cv2.waitKey(0)
	
	################################### MODELS ################################
	GEN = GeneratorModel()
	DISC = DiscriminatorModel()
	D_ON_G = DISC_ON_GEN(GEN, DISC)
	
	GEN_OPT = RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0)
	DISC_OPT = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
	
	DISC.compile(loss=WGAN_LOSS_DISC, optimizer=DISC_OPT)
	GEN.compile(loss='binary_crossentropy', optimizer=GEN_OPT)		# BCE loss connects models and seems to work well for most data including images
	D_ON_G.compile(loss=WGAN_LOSS_GEN, optimizer=GEN_OPT)			# TODO - does loss make sense? 
	
	for layer in D_ON_G.layers:
		print(layer.name + " /// " + str(layer.output_shape))
	
	
	#print("--------------------------")
	#for layer in GEN.layers:
	#	print(layer.name + " /// " + str(layer.output_shape))
		
	
	
	'''
	ViewFrame4(frame_test)
	
	ms1 = time.time()*1000.0
	f1 = PreprocessImgFromDatastore(frame_test[:,:,0:3])
	print(f1.shape)
	vgg_feature_test = VGGFeatures(VGG_model, f1)
	ms2 = time.time()*1000.0
	print('Time to get VGG features (block2_pool) from 128x128x3 frame:' + str(round(ms2-ms1,4)) + ' ms')
	
	print(vgg_feature_test.shape)
	
	test = vgg_feature_test[0,:,:,0]
	print(test.shape)

	print(test)
	cv2.imshow('test', cv2.resize(test, (128,128)))
	cv2.waitKey(0)
	'''
	
	print("\n---------- SEQUENTIAL CONTROL GAN TRAINING ----------\n")
	
	itrs = 0
	maxDiscFactor = 0.9 # only train GEN when at this percent of max Disc loss
	while itrs < TOTAL_GANS_TRAINING_ITRS:
	
		# Choose which expert demonstration we're using: 
		CURR_EXPERT_DEMONSTRATION = randint(0,NUM_EXPERT_DEMONSTRATIONS-1)

		frames, actions = LoadFramesActionsFromFolder(DEMONSTRATION_FOLDERS[CURR_EXPERT_DEMONSTRATION])	# about 5 secs
		LENGTH_CURR_DEMONSTRATION = frames.shape[0] # number of timesteps for this demonstration
		
		
		# New demonstration, reset the relevant buffers:
		# For GENERATOR input
		PREV_FRAMES_BUFFER = np.zeros((NUM_PAST_FRAMES, CAM_W, CAM_H, CAM_C))
		PREV_ACTION_BUFFER = np.zeros((NUM_PAST_FRAMES, ACTION_LEN))
		# For DISCRIMINATOR input
		FUTURE_FRAMES_BUFFER = np.zeros((NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))
		FUTURE_ACTION_BUFFER = np.zeros((NUM_FUTURE_FRAMES, ACTION_LEN))
		
		for i in range(0, round(LENGTH_CURR_DEMONSTRATION/NUM_FUTURE_FRAMES)): 
			# only sample a limited number from each demo before moving on to next demo. 
			t = randint(0,LENGTH_CURR_DEMONSTRATION-1)	# now we have random shuffle training within a demo. 
			
			# Generator: 
			# - get past frames
			# - get past actions
			# - get noise
			# - get current frame
			# recall: model_inputs = [input_curr_frame, input_prev_frames_raw, input_prev_actions_raw, input_noise]
			ms1 = time.time()*1000.0
			
			# CURRENT FRAME, ACTION
			input_curr_frame = frames[t,:] # 128 x 128 x 12
			#curr_action = actions[t,:] --> this is ultimately what we want
			
			if (t >= NUM_PAST_FRAMES): # regular, in-bounds
				PREV_FRAMES_BUFFER = frames[t-NUM_PAST_FRAMES:t, :]
				PREV_ACTION_BUFFER_BUFFER = actions[t-NUM_PAST_FRAMES:t, :]
			else: # t is less than past frames, need to concat zeros at beginning
				LIM1 = np.abs(t - NUM_PAST_FRAMES) 
				PREV_FRAMES_BUFFER = np.concatenate((np.zeros((LIM1, CAM_W, CAM_H, CAM_C)), frames[0:t, :]), axis=0) 
				PREV_ACTIONS_BUFFER = np.concatenate((np.zeros((LIM1, ACTION_LEN)), actions[0:t, :]), axis=0)

			if (t <= LENGTH_CURR_DEMONSTRATION - NUM_FUTURE_FRAMES - 1):
				FUTURE_FRAMES_BUFFER = frames[t+1:t+NUM_FUTURE_FRAMES+1, :]
				FUTURE_ACTIONS_BUFFER = actions[t:t+NUM_FUTURE_FRAMES, :]
			else: # running off the end
				LIM2 = t + NUM_FUTURE_FRAMES - LENGTH_CURR_DEMONSTRATION + 1 # we need to repeat last frame this many times / add this many zero actions
				FUTURE_FRAMES_BUFFER = np.concatenate((frames[t+1:LENGTH_CURR_DEMONSTRATION, :], np.zeros((LIM2, CAM_W, CAM_H, CAM_C))), axis=0) 
				FUTURE_ACTIONS_BUFFER = np.concatenate((actions[t:LENGTH_CURR_DEMONSTRATION, :], np.zeros((LIM2-1, ACTION_LEN))), axis=0) 
			
			# NOISE INPUT
			input_noise = CLIP*np.random.randn(NOISE_DIM,)
			
			ms2 = time.time()*1000.0
			GEN_OUTPUTS = GEN.predict([np.expand_dims(input_curr_frame, axis=0), 
									   np.expand_dims(PREV_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_PAST_FRAMES), axis=0), 
									   np.expand_dims(PREV_ACTION_BUFFER.reshape(-1), axis=0), 
									   np.expand_dims(input_noise, axis=0)])
									   
			GEN_FUTURE_FRAMES = GEN_OUTPUTS[0]
			GEN_FUTURE_ACTIONS = GEN_OUTPUTS[1]
			if ((itrs-1) % DISC_PER_GEN_ITRS == 0 and itrs > INITIAL_DISC_ITRS):		# i.e. right after Gen update
				timestamp = str(time.time()).replace(".","")
				imgs_gen = ViewFutureFrames(GEN_FUTURE_FRAMES.reshape(NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))
				imgs_split = np.zeros((30, NUM_FUTURE_FRAMES*CAM_W, 3)) # TODO: put this inbetween
				imgs_gt = ViewFutureFrames(FUTURE_FRAMES_BUFFER)
				#print(imgs_gt.shape); print(imgs_split.shape); print(imgs_gen.shape);
				imgs_compare = np.concatenate((imgs_gt, imgs_split, imgs_gen),axis=0)
				#print(imgs_compare.shape)
				#cv2.imshow('image',imgs_compare) 
				#cv2.waitKey(0)
				cv2.imwrite('sample_'+timestamp+'.png',(imgs_compare*255))
				
			ms3 = time.time()*1000.0
			
			
			'''print(input_curr_frame.shape); print("\n---")
			print(PREV_FRAMES_BUFFER.shape); print("\n---")
			print(PREV_ACTION_BUFFER.shape); print("\n---")
			print(FUTURE_FRAMES_BUFFER.shape); print("\n---")
			print(FUTURE_ACTION_BUFFER.shape); print("\n---")
			print(GEN_FUTURE_FRAMES.shape); print("\n---")
			print(GEN_FUTURE_ACTIONS.shape); print("\n---")'''
			
			GEN.trainable = False 
			DISC.trainable = True
			D_of_fake = DISC.predict([GEN_FUTURE_FRAMES,GEN_FUTURE_ACTIONS])
									  
			# loss function: multiply output of DISC with target, either -1 or 1
			Disc_Loss = DISC.train_on_batch([np.expand_dims(FUTURE_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_PAST_FRAMES), axis=0), 
										     np.expand_dims(FUTURE_ACTION_BUFFER.reshape(-1), axis=0)], # REAL
										     np.array([D_of_fake[0]])) # last array: outputs (decrease: D(real)*(-1))
			
			ms4 = time.time()*1000.0
			print("===================================================")
			print("iter: " + str(itrs)); print("\n---")
			print("timestep t: " + str(t)); print("\n---")
			print('Data setup time: ' +str(round((ms2-ms1)/1000, 2)) + ' secs')
			print('GEN compute time: ' +str(round((ms3-ms2)/1000, 2)) + ' secs')
			print('DISC update time: ' +str(round((ms4-ms3)/1000, 2)) + ' secs')
			print("DISC LOSS and D_of_fake: ")
			print(str(Disc_Loss) + " // " + str(D_of_fake[0][0]) + "\n")
			DISC_LOSS_BUFFER = np.roll(DISC_LOSS_BUFFER, 1); DISC_LOSS_BUFFER[0] = Disc_Loss
			curr_mean = np.mean(DISC_LOSS_BUFFER)
			if (curr_mean <= DISC_LOSS_MEAN_MAX):
				DISC_LOSS_MEAN_MAX = curr_mean
			print(DISC_LOSS_BUFFER); print("\n")
			print("DISC LOSS BUFFER Max (mean): " + str(round(DISC_LOSS_MEAN_MAX,3)))
			print("Actual future actions a(t)...a(t+NFF-1):"); print(FUTURE_ACTIONS_BUFFER)
			print("Predicted future actions a(t)...a(t+NFF-1):"); print(GEN_FUTURE_ACTIONS[0].reshape(NUM_FUTURE_FRAMES, ACTION_LEN))
								
			 # Clip discriminator weights after gradient updates: 
			for l in DISC.layers:
				weights = l.get_weights()
				weights = [np.clip(w, -CLIP, CLIP) for w in weights]
				l.set_weights(weights)
				
			############################ GENERATOR TRAINING ##########################
			if 	(itrs % DISC_PER_GEN_ITRS == 0 and itrs > INITIAL_DISC_ITRS and (Disc_Loss <= maxDiscFactor*DISC_LOSS_MEAN_MAX)):
				print("---------- Generator Training Iteration -----------")
				ms5 = time.time()*1000.0
				GEN.trainable = True
				DISC.trainable = False
				
				input_noise = CLIP*np.random.randn(NOISE_DIM,) # re-compute noise for novel input to Disc
				Gen_Loss = D_ON_G.train_on_batch([np.expand_dims(input_curr_frame, axis=0), 
												  np.expand_dims(PREV_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_PAST_FRAMES), axis=0), 
									              np.expand_dims(PREV_ACTION_BUFFER.reshape(-1), axis=0), 
									              np.expand_dims(input_noise, axis=0)],
												  np.array([0])) # Not using y_true value
				ms6 = time.time()*1000.0
				print('GEN update time: ' +str(round((ms6-ms5)/1000, 2)) + ' secs')
				print("GEN LOSS: ")
				print(str(Gen_Loss) +"\n")
				GEN_LOSS_BUFFER = np.roll(GEN_LOSS_BUFFER, 1); GEN_LOSS_BUFFER[0] = Gen_Loss
				print(GEN_LOSS_BUFFER); print("\n")
				
				
			#imgs = ViewFutureFrames(FUTURE_FRAMES_BUFFER[0:10,:])
			#cv2.imshow('image',imgs)
			#cv2.waitKey(1)
	
			if (itrs % SAVE_CHECKPOINT_ITRS == 0 and itrs > INITIAL_DISC_ITRS):
				timestamp = str(time.time()).replace(".","")
				disc_str_name = timestamp+'_DISC_'+'.h5' # pro-tip: manually re-name after each run... 
				DISC.save(disc_str_name)
				gen_str_name = timestamp+'_GEN_'+'.h5' # pro-tip: manually re-name after each run... 
				GEN.save(gen_str_name)
	
			itrs = itrs + 1
	
	#print("\n---------- STAGE 2: ADVERSARIAL TRAINING ----------\n")
	#for itrs in range(0,TOTAL_GANS_TRAINING_ITRS):
	#
	#	print("asdf")
	
	
	
	
	
	
	
	