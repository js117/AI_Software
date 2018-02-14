# STACK: - Anaconda3 (Python 3.5.2, Anaconda 4.2.0 (64-bit): C:\Program Files\Anaconda3
#		 - TensorFlow backend (see: C:\Users\JDS\.keras\keras.json)
#		 - pip install --upgrade --ignore-installed tensorflow

import time
ms1 = 0; ms2 = 0
print("\n\n --- Loading ML libraries (about 10 secs)... --- \n\n\n")
ms1 = time.time()*1000.0
import tensorflow as tf
import numpy as np
import cv2
import os.path
import sys
from sys import platform
from random import randint
from math import sqrt
ms2 = time.time()*1000.0
print("\nLoaded ML libraries. Time elasped: ", int(ms2-ms1), " ms\n")

# Project constants: file system, video settings
MODEL_STRING_NAME = 'my_model.h5'
VIDEO_FRAMES_FOLDER_NAME = 'sequence_video_frames'
OUTPUT_FOLDER_NAME = 'output_frames'
MODEL_CHECKPOINT_NAME = 'VNFP_CLSTM'
VIDEO_NAME = 'sample_sequence_downsampled.mp4'
WIDTH = 112 
HEIGHT = 112
NUM_CHANNELS = 3
global TIMESTEPS
TIMESTEPS = 30
minus1 = (np.array([-1]).astype('float32'))[0] # JUST

# View an image, resized: 
#cv2.namedWindow("img1", cv2.WINDOW_AUTOSIZE); cv2.imshow("img1", cv2.resize(cv2.imread(img_path1), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img2", cv2.WINDOW_AUTOSIZE); cv2.imshow("img2", cv2.resize(cv2.imread(img_path2), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img3", cv2.WINDOW_AUTOSIZE); cv2.imshow("img3", cv2.resize(cv2.imread(img_path3), (224, 224))); cv2.waitKey(0)

### TEST PLAYING VIDEO, create frame database if not exist already: 

def PrepareVideoFilesForTraining(vid_name, vid_folder):
# e.g. vid_name = 'sample_sequence_downsampled.mp4'
#	   vid_folder = 'sequence_video_frames'
	if not os.path.exists(vid_folder):
		os.makedirs(vid_folder)

		cap = cv2.VideoCapture(vid_name)
		numFrames = 0
		while True:
			if cap.grab():
				flag, frame = cap.retrieve()
				numFrames = numFrames + 1
				#name = "%d.jpg"%numFrames
				name = "%d.jpg"%(time.time()*1000)
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

	# Create training and testing data: approx. 70 - 30 
	curr_dir = os.getcwd()
	search_dir = ""
	if platform == "win32":
		search_dir = os.getcwd()+"\\"+VIDEO_FRAMES_FOLDER_NAME # WINDOWS
	else:
		search_dir = os.getcwd()+"/"+VIDEO_FRAMES_FOLDER_NAME # LINUX
	os.chdir(search_dir)
	files = filter(os.path.isfile, os.listdir(search_dir))
	files = [os.path.join(search_dir, f) for f in files] # add path to each file
	files.sort(key=lambda x: x) #os.path.getmtime(x))
	os.chdir(curr_dir)
	
	return files
	# Now our files are sorted chronologically (they were created in this order)
	# and we can put into training data, similar to below format:
	#(60000, 28, 28, 1) # train (num_examples, width, height, channels)
	#(10000, 28, 28, 1) # test (num_examples, width, height, channels)
	

def CreateVidFrameTensor(files, width, height):
# Create and return a tensor of video frame files that we can use in some neural network model
	X_ALL = []
	itr = 0
	print("Preparing video files for training...")
	for file in files:
		itr = itr + 1
		im = cv2.resize(cv2.imread(file), (width, height))
		#s = im.shape
		#print(s)
		#im = np.reshape(im, (s[2], s[0], s[1]))
		X_ALL.append(im) #default: WIDTH, HEIGHT, NUM_CHANNELS
		
	X_ALL = np.asarray(X_ALL); 
	X_ALL = X_ALL.astype('float32') / 255.
	print("Video files prepared. Shape: ")		
	print(X_ALL.shape)
	NumFrames = X_ALL.shape[0]
	
	return X_ALL, NumFrames
	
# Pass in the session and the graph output xp (== prediction)
def Generate_Test_Output_Video(X_ALL, sess, xp, x):
	global TIMESTEPS
	print("Generating output test video...\n")
	num_frames = X_ALL.shape[0]
	prev_frame = np.array([X_ALL[0,:,:,:]]) # to make shapes work initially.. 
	RECREATED_FRAMES = np.zeros(X_ALL.shape)

	print("\n Preparing sample output video...\n")
	NUM_FRAMES_RESET = TIMESTEPS # related to the training # of timesteps
	TIMESTEPS = 1 # this is key: we're not unrolling the LSTM since we're predicting, not training. 

	fake_batch = np.zeros([NUM_FRAMES_RESET+1, WIDTH, HEIGHT, NUM_CHANNELS]) 
	
	for i in range(1, num_frames): 
			
		fake_batch[0] = prev_frame
		xpreds = sess.run([xp], feed_dict={x: fake_batch })
		print(":::DEBUG:::"+str(i)+" /// "+str(xpreds[0][0].shape)+"\n")
		
		RECREATED_FRAMES[i] = xpreds[0][0]
		
		# ADD A RESET: since we degrade quite a bit, let's reset by using an actual frame
		# Mathematically: using x(t+1) ~ p( x(t+1) | x(t) ) will degrade over time, hence the reset 
		if i%NUM_FRAMES_RESET == 0:
			prev_frame = np.array([X_ALL[i,:,:,:]])
		else:
			prev_frame = xpreds[0][0]
			
	# Now make a lil video: 
	print(":::DEBUG:::"+str(RECREATED_FRAMES.shape)+"\n")

	# Get ready to create output videos: 
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output_'+str(time.time())+'.avi',fourcc, 8.0, (2*WIDTH,HEIGHT))
		
	print("Creating comparison video: ")

	if not os.path.exists(OUTPUT_FOLDER_NAME):
		os.makedirs(OUTPUT_FOLDER_NAME)

	for i in range(0, num_frames):
		img1 = X_ALL[i,:,:,:]
		img2 = RECREATED_FRAMES[i,:,:,:]
		vis = np.concatenate((img1, img2), axis=1)
		#cv2.imshow('video', vis)
		#cv2.waitKey(100) # play at ~10 fps
		the_img = (vis * 255.0).astype('u1')
		out.write(the_img) # don't forget actual videos can't be b/w 0 and 1

		# very hacky... _fourcc stuff isn't working on my linux machine... 
		cv2.imwrite(OUTPUT_FOLDER_NAME+"/"+str(time.time()*1000)+".jpg", the_img)		
	
	# Release everything if job is finished
	out.release()
	cv2.destroyAllWindows()
	
################################################### MODEL-SPECIFIC #########################################################	

# Utility functions	:
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def conv2d_add_sigmoid(x1, W1, x2, W2, b, strides=1):
    # Custom function as needed for CLSTMs
	t1 = tf.nn.conv2d(x1, W1, strides=[1, strides, strides, 1], padding='SAME')
	t2 = tf.nn.conv2d(x2, W2, strides=[1, strides, strides, 1], padding='SAME')
	t3 = tf.add(t1, t2)
	t3 = tf.nn.bias_add(t3, b)
	return tf.nn.sigmoid(t3)
def conv2d_add_tanh(x1, W1, x2, W2, b, strides=1):
    # Custom function as needed for CLSTMs
	t1 = tf.nn.conv2d(x1, W1, strides=[1, strides, strides, 1], padding='SAME')
	t2 = tf.nn.conv2d(x2, W2, strides=[1, strides, strides, 1], padding='SAME')
	t3 = tf.add(t1, t2)
	t3 = tf.nn.bias_add(t3, b)
	return tf.nn.tanh(t3)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	
def binary_crossentropy_loss_from_probs(z, x):
	# inputs are both tensors of the same size
	# targets = z, predictions = x
	# then: return (-z*log(x) - (1-z)*log(1-x))
	t1 = tf.multiply(z, tf.log(tf.clip_by_value(x,1e-10,1.0)))
	t2 = tf.multiply(1.0 - z, tf.log(tf.clip_by_value(1.0 - x,1e-10,1.0)))
	return (tf.multiply(minus1, t1) + tf.multiply(minus1, t2))
	
# WEIGHTS / Params: 
CONV_STRIDES = 5
NUM_OUT_CONV1 = 8
NUM_OUT_CONV2 = 16
NUM_OUT_CONV3 = 32			# up to here: the encoder params
NUM_OUT_CONV_LSTM_1 = 32 	# first idea: just match the input
NUM_OUT_CONV_LSTM_2 = 32 
NUM_OUT_DECONV1 = 16		# deconv = decoder params (below here)
NUM_OUT_DECONV2 = 8
NUM_OUT_DECONV3 = 3 # equals NUM_CHANNELS

# For convolutional layers, there are "num input feature maps * filter height * filter width" inputs to each hidden unit
faninE_WC1 = NUM_CHANNELS * CONV_STRIDES * CONV_STRIDES 
faninE_WC2 = NUM_OUT_CONV1 * CONV_STRIDES * CONV_STRIDES
faninE_WC3 = NUM_OUT_CONV2 * CONV_STRIDES * CONV_STRIDES 
faninCLSTM1_ifmm = max(NUM_OUT_CONV3, NUM_OUT_CONV_LSTM_1)  # input feature map max input number - either "from below" or "from before"
faninCLSTM1 = faninCLSTM1_ifmm * CONV_STRIDES * CONV_STRIDES
faninCLSTM2_ifmm = max(NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_2)  # input feature map max input number - either "from below" or "from before"
faninCLSTM2 = faninCLSTM2_ifmm * CONV_STRIDES * CONV_STRIDES
faninD_WC1 = NUM_OUT_CONV_LSTM_2 * CONV_STRIDES * CONV_STRIDES
faninD_WC2 = NUM_OUT_DECONV1 * CONV_STRIDES * CONV_STRIDES 
faninD_WC3 = NUM_OUT_DECONV2 * CONV_STRIDES * CONV_STRIDES 

with tf.name_scope("weights_CNN_Encoder"):
	weights_CNN_Encoder = {
		# e.g. 5x5 conv, 3 input channels, 8 outputs
		'WC1': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_CHANNELS, NUM_OUT_CONV1],  dtype=tf.float32) / sqrt(faninE_WC1)), # e.g. [5, 5, 3, 16]
		# e.g. 5x5 conv, 8 input channels, 16 outputs
		'WC2': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV1, NUM_OUT_CONV2], dtype=tf.float32) / sqrt(faninE_WC2)),
		# e.g. 5x5 conv, 16 input channels, 32 outputs
		'WC3': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV2, NUM_OUT_CONV3], dtype=tf.float32) / sqrt(faninE_WC3))
	}
	biases_CNN_Encoder = {
		'BC1': tf.Variable(tf.random_normal([NUM_OUT_CONV1], dtype=tf.float32) / sqrt(NUM_CHANNELS)),
		'BC2': tf.Variable(tf.random_normal([NUM_OUT_CONV2], dtype=tf.float32) / sqrt(NUM_OUT_CONV1)),
		'BC3': tf.Variable(tf.random_normal([NUM_OUT_CONV3], dtype=tf.float32) / sqrt(NUM_OUT_CONV2))
	}

# TODO with LSTM weights: use orthogonal matrices to init? 
# see: https://github.com/Lasagne/Lasagne/blob/a3d44a7fbb84b1206a3959881c52b2203f48fc44/lasagne/init.py#L363 
with tf.name_scope("weights_CLSTM1"):
	weights_CLSTM1 = {
		# First Conv LSTM layer:
		# Input: 
		'WC_Xi': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV3, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)),	    # applies to input "from below" (from encoder)
		'WC_Hi': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)), # applies to input "from before"
		# Forget Gate:
		'WC_Xf': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV3, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)),		# applies to input "from below" (from encoder)
		'WC_Hf': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)), # applies to input "from before"
		# Output Gate:
		'WC_Xo': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV3, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)),		# applies to input "from below" (from encoder)
		'WC_Ho': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)), # applies to input "from before"
		# Input Gate:
		'WC_Xg': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV3, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)),		# applies to input "from below" (from encoder)
		'WC_Hg': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1)), # applies to input "from before"
	}
	biases_CLSTM1 = {
		# First Conv LSTM layer:
		# Input:
		'BC_i': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1_ifmm)),
		# Forget Gate:
		'BC_f': tf.Variable(tf.ones([NUM_OUT_CONV_LSTM_1], dtype=tf.float32) ),
		# Output Gate:
		'BC_o': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1_ifmm)),
		# Input Gate:
		'BC_g': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_1], dtype=tf.float32) / sqrt(faninCLSTM1_ifmm))
	}
	
with tf.name_scope("weights_CLSTM2"):	
	weights_CLSTM2 = {
		# Second Conv LSTM layer:
		# Input: 
		'WC_Xi': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from below" (from CLSTM1)
		'WC_Hi': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_2, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from before"
		# Forget Gate:
		'WC_Xf': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from below" (from CLSTM1)
		'WC_Hf': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_2, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from before"
		# Output Gate:
		'WC_Xo': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from below" (from CLSTM1)
		'WC_Ho': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_2, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from before"
		# Input Gate:
		'WC_Xg': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_1, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2)), # applies to input "from below" (from CLSTM1)
		'WC_Hg': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_2, NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2))  # applies to input "from before"
	}
	biases_CLSTM2 = {
		# First Conv LSTM layer:
		# Input:
		'BC_i': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2_ifmm)),
		# Forget Gate:
		'BC_f': tf.Variable(tf.ones([NUM_OUT_CONV_LSTM_2], dtype=tf.float32) ),
		# Output Gate:
		'BC_o': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2_ifmm)),
		# Input Gate:
		'BC_g': tf.Variable(tf.random_normal([NUM_OUT_CONV_LSTM_2], dtype=tf.float32) / sqrt(faninCLSTM2_ifmm))
	}

with tf.name_scope("weights_CNN_Decoder"):
	weights_CNN_Decoder = {
		'WC1': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_CONV_LSTM_2, NUM_OUT_DECONV1], dtype=tf.float32) / sqrt(faninD_WC1)), 
		'WC2': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_DECONV1, NUM_OUT_DECONV2], dtype=tf.float32) / sqrt(faninD_WC2)),
		'WC3': tf.Variable(tf.random_normal([CONV_STRIDES, CONV_STRIDES, NUM_OUT_DECONV2, NUM_CHANNELS], dtype=tf.float32) / sqrt(faninD_WC3))
	}
	biases_CNN_Decoder = {
		'BC1': tf.Variable(tf.random_normal([NUM_OUT_DECONV1], dtype=tf.float32) / sqrt(NUM_OUT_CONV_LSTM_2)),
		'BC2': tf.Variable(tf.random_normal([NUM_OUT_DECONV2], dtype=tf.float32) / sqrt(NUM_OUT_DECONV1)),
		'BC3': tf.Variable(tf.random_normal([NUM_CHANNELS], dtype=tf.float32) / sqrt(NUM_OUT_DECONV2))
	}

def ResetCLSTMStates(): # Assumes variables are GLOBAL
	# CLSTM Layer 1:
	h_prev_1 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_1], dtype=np.float32) 
	c_prev_1 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_1], dtype=np.float32) 
	# CLSTM Layer 2:
	h_prev_2 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_2], dtype=np.float32) 
	c_prev_2 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_2], dtype=np.float32) 
	
			

# Model Architecture:
# x_true(p) --> CNN_Encoder --> Flatten --> Recurrent Layer(s) --> CNN_Decoder --> x_pred(p+1)
# where x(p) is a sequential batch of images
# Loss: from comparing the images x_pred(p+1) to x_true(p+1)
# Recurrent Layers implemented as Conv LSTMs, see https://arxiv.org/pdf/1506.04214v1.pdf 
   			
def CNN_Encoder(x, weights, biases):

	x = tf.reshape(x, shape=[-1, WIDTH, HEIGHT, NUM_CHANNELS])

    # Convolution Encoder Layers
	C1 = conv2d(x, weights['WC1'], biases['BC1'])
	C2 = conv2d(C1, weights['WC2'], biases['BC2'])
	C3 = conv2d(C2, weights['WC3'], biases['BC3'])
	# At this point, we have: 112*112*3*32 = 1204224 numbers in our output activation map. 
	# We cannot flatten and work with this unfortunately, at least not on smaller CPU
	# Doing Convolutional LSTMs: add the output tensors ? 
	
	return C3 # example output dims: 112,112,32
	
def CLSTM_Layer(x, h_prev, c_prev, weights, biases): # will take in the above, and saved states
# Pass in weights_CLSTM1 and biases_CLSTM1 for CLSTM layer 1, etc.
# Internal variable naming convention must be consistent, e.g. 'WC_Xi', 'WC_Ho', 'BC_g', etc.
	
	input = conv2d_add_sigmoid(x, weights['WC_Xi'], h_prev, weights['WC_Hi'], biases['BC_i'])
	forget = conv2d_add_sigmoid(x, weights['WC_Xf'], h_prev, weights['WC_Hf'], biases['BC_f'])
	output = conv2d_add_sigmoid(x, weights['WC_Xo'], h_prev, weights['WC_Ho'], biases['BC_o'])
	input_gate = conv2d_add_tanh(x, weights['WC_Xg'], h_prev, weights['WC_Hg'], biases['BC_g'])
	
	c_curr = tf.add(tf.multiply(forget, c_prev), tf.multiply(input, input_gate)) # c(t) = f .* c(t-1) + i .* g
	h_curr = tf.multiply(output, tf.nn.tanh(c_curr))

	return h_curr, c_curr
	
def CNN_Decoder(x, weights, biases):	# will take in something of size (WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_{last_layer}) e.g. (112, 112, 32)

	# Convolution Decoder Layers; will go from 32 (CLSTM2 layer) to 16, to 8, and finally to 3 channels 
	C1 = conv2d(x, weights['WC1'], biases['BC1'])
	C2 = conv2d(C1, weights['WC2'], biases['BC2'])
	C3 = conv2d(C2, weights['WC3'], biases['BC3'])
	
	return C3
	
# HOW TO TRAIN: start with state variables (h, c) as zeros; reset them to zero after a complete sweep through sequential data.
# Reference: https://gist.github.com/karpathy/d4dee566867f8291f086 	
def MODEL_AND_LOSS_ON_SEQ(x, h_prev_1, c_prev_1, h_prev_2, c_prev_2): 
# x inputted as 4D tensor of shape (TIMESTEPS, WIDTH, HEIGHT, NUM_CHANNELS) of sequential 3D frames
# xp - placeholder to take the predictions so we can visualize 
# h_prev_i - single 3D tensors for the CLSTM layers
# c_prev_i - ^ same (for internal states)

	cost = 0
	xp_list = []
	
	for t in range(0, TIMESTEPS):
		# As we iterate over TIMESTEPS, we change the input and target, while also updating LSTM state variables.
		# The caller of this function determines resetting the state variables ("stateful" or not)
		training_input = tf.reshape(x[t,:,:,:], [-1,WIDTH,HEIGHT,NUM_CHANNELS])	# will give input image of (1, WIDTH, HEIGHT, NUM_CHANNELS) dims --> conv2d needs 4d tensor
		training_target = tf.reshape(x[t+1,:,:,:], [-1,WIDTH,HEIGHT,NUM_CHANNELS]) # outer loop calling this fnc must ensure proper limiting based on TIMESTEPS
		
		enc = CNN_Encoder(training_input, weights_CNN_Encoder, biases_CNN_Encoder)
		
		h_prev_1, c_prev_1 = CLSTM_Layer(enc, h_prev_1, c_prev_1, weights_CLSTM1, biases_CLSTM1)
		
		#h_prev_2, c_prev_2 = CLSTM_Layer(h_prev_1, h_prev_2, c_prev_2, weights_CLSTM2, biases_CLSTM2)
		
		prediction = CNN_Decoder(h_prev_1, weights_CNN_Decoder, biases_CNN_Decoder)
		
		xp_list.append(prediction[0,:,:,:])
		
		cost = cost + tf.reduce_mean(binary_crossentropy_loss_from_probs(training_target, prediction)) 
		
	xpreds = tf.pack(xp_list)
	
	return [cost, xpreds]
		
# Parameters
global learning_rate
# ^ HOW TO MODIFY WHILE RUNNING: open new terminal (in folder), and type:
# echo 0.005 > lr.txt 
# e.g. some other learning rate number. 
training_iters = 200000
lr_default = 0.0001
lr_decrease_schedule_stp = 60000 # every this many steps, decrease LR by some amount
lr_decrease_schedule_amt = 0.1 # make it this fraction of its previous LR
display_step = 1
save_step = 100
# CLSTM State and Internal Variables: 
# CLSTM Layer 1 Hidden and Internal States:
global h_prev_1; global c_prev_1; #global xp
#xp = np.zeros([TIMESTEPS+1, WIDTH, HEIGHT, NUM_CHANNELS])
h_prev_1 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_1], dtype=np.float32) # first dim (1) should match our fake batch size
c_prev_1 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_1], dtype=np.float32) 
# CLSTM Layer 2 Hidden and Internal States:
global h_prev_2; global c_prev_2
h_prev_2 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_2], dtype=np.float32) 
c_prev_2 = np.zeros([1, WIDTH, HEIGHT, NUM_OUT_CONV_LSTM_2], dtype=np.float32) 

def TRAIN_FULL_MODEL(VidFrames, NumFrames):
	
	# our default multiplied by a decay factor based on num steps in previous run.
	learning_rate = lr_default * lr_decrease_schedule_amt ** round(LOAD_PREV_MODEL_STEPS / lr_decrease_schedule_stp)
	
	# tf Graph input
	# Video Frames, created into inputs and targets in MODEL function 
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [TIMESTEPS+1, WIDTH, HEIGHT, NUM_CHANNELS], name="video-input-sequence")
	with tf.name_scope('learning_rate'):
		learning_rate_tf = tf.placeholder(tf.float32, shape=[])
		
	# Construct model, loss, optimizer:
	main_model_function = MODEL_AND_LOSS_ON_SEQ(x, h_prev_1, c_prev_1, h_prev_2, c_prev_2)
	with tf.name_scope('cost'):
		cost = main_model_function[0] #MODEL_AND_LOSS_ON_SEQ(x, h_prev_1, c_prev_1, h_prev_2, c_prev_2)
	with tf.name_scope('optimizer'):	
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_tf).minimize(cost)
	with tf.name_scope('predictions'):
		xp = main_model_function[1]
		#xp = np.zeros([TIMESTEPS, WIDTH, HEIGHT, NUM_CHANNELS]) # will get xp_list from cost
		#xp = np.zeros([TIMESTEPS+1, WIDTH, HEIGHT, NUM_CHANNELS])
		#xp = tf.zeros([TIMESTEPS, WIDTH, HEIGHT, NUM_CHANNELS], dtype=tf.float32, name="video-pred-sequence")
		#xp = tf.placeholder(tf.float32,[TIMESTEPS+1, WIDTH, HEIGHT, NUM_CHANNELS], name="video-pred-sequence")
		
	with open('lr.txt', 'w') as f:
		f.write(str(learning_rate))
	f.close()
		
	tf.scalar_summary("cost", cost)
	tf.image_summary("inputs", x, max_images=TIMESTEPS)
	tf.image_summary("predictions", xp, max_images=TIMESTEPS)
	tf.image_summary("encoder_firstlvl_filters", tf.reshape(weights_CNN_Encoder['WC1'], [NUM_OUT_CONV1,CONV_STRIDES,CONV_STRIDES,NUM_CHANNELS]), max_images=NUM_OUT_CONV1)
	
	# TensorBoard stuff: 
	summary_op = tf.merge_all_summaries()
	
	# Initializing the variables
	init = tf.initialize_all_variables()
	
	# Ready the saver:
	saver = tf.train.Saver()
	
	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		
		# TensorBoard stuff: 
		writer = tf.summary.FileWriter('logs', sess.graph)
		# Run the following in terminal before launching script: tensorboard --logdir=logs
		
		step = LOAD_PREV_MODEL_STEPS + 1 # defaults at 1 unless changed
		seek = 0
		lim = NumFrames-TIMESTEPS-2	# this ensures we can loop 0 to TIMSTEPS and still access x[t+1,:,:,:]
		# Keep training until reach max iterations
		
		if (LOAD_PREV_MODEL != ""): # After creating graph, can load model and restore weights/variables.  
			#new_saver = tf.train.import_meta_graph(LOAD_PREV_MODEL)
			#new_saver.restore(sess, tf.train.latest_checkpoint('./')) # assuming model files in same main folder
			ckpt = tf.train.get_checkpoint_state(os.getcwd())
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("\nMODEL RESTORED!"+str(ckpt)+"\n")
			else:
				print("...no checkpoint found...")
		
			if (FLAG_GENERATE_VIDEO == 1):
				Generate_Test_Output_Video(VidFrames, sess, xp, x) # Pass in the session and the graph output xp (== prediction)
				print("Finished generating video. See output_[timestamp].avi\n")
				return
			
		
		while step < training_iters:
			ms1 = time.time()*1000.0
			
			with open('lr.txt', 'r') as f:
				learning_rate = float(f.readline())
			if learning_rate > 10 or learning_rate < 0: 
				print("Warning - invalid learning rate found in lr.txt. Resetting to default scheduled value. ")
				learning_rate = lr_default * lr_decrease_schedule_amt ** round(LOAD_PREV_MODEL_STEPS / lr_decrease_schedule_stp)
				with open('lr.txt', 'w') as f:
					f.write(str(learning_rate))
				f.close()

			#seek = randint(0, lim) # inclusive
			seek = seek + 1 # round(TIMESTEPS/2)
			if seek >= lim: 
				seek = 0
				ResetCLSTMStates()
				
			batch = VidFrames[seek:seek+TIMESTEPS+1,:,:,:]
			
			_, loss, xpreds, summary = sess.run([optimizer, cost, xp, summary_op], feed_dict={x: batch, learning_rate_tf: learning_rate})
			
			#print("\n\nDEBUG::: "+str(xpreds[0].shape))
			
			# write log
			writer.add_summary(summary, step)
			ms2 = time.time()*1000.0
			
			if step % lr_decrease_schedule_stp == 0:
				learning_rate = learning_rate * lr_decrease_schedule_amt
				with open('lr.txt', 'w') as f:
					f.write(str(learning_rate))
				f.close()
			
			if step % display_step == 0:
            # Calculate batch loss
				summary_str = "Iter " + str(step) + " / seek: " + str(seek) + " / lr: " + str(learning_rate) + ", Current Loss= " + "{:.6f}".format(loss) + " /// Time Elapsed: " + str(int(ms2-ms1)) + "\n"
				print(summary_str)
				with open('manual_log.txt','a') as g:
					g.write(summary_str)
					if step % (display_step+30) == 0:
						# some manual checks of the images to see if mean pixels are getting closer to targets
						temp1 = np.mean(batch[0]); temp2 = np.sum(batch[0])
						temp3 = np.mean(xpreds[0]); temp4 = np.sum(xpreds[0])			# xpreds should be a list
						print("\n\nMEAN PIXEL CHECK: input_test: "+str(temp1)+"///"+str(temp2)+"\n")
						print("\n\nMEAN PIXEL CHECK: pred__test: "+str(temp3)+"///"+str(temp4)+"\n\n")

			if step % save_step == 0:
				print("\n===== MODEL SAVE CHECKPOINT =====\n")
				save_path = MODEL_CHECKPOINT_NAME
				if not os.path.isabs(save_path):
					save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
				saver.save(sess, save_path, global_step=step)
				
			step = step + 1
		
	saver.save(sess, MODEL_CHECKPOINT_NAME+'_completed', global_step=step)
	print("Optimization Finished!")
		
# TODO: 
# x figure out shapes of h_prev, c_prev[]
# x create wrapper and training loop for MODEL_AND_LOSS_ON_SEQ
# x create save checkpoints 
# x TEST existing: make code run and train something
# x make sensible initializations for the LSTM cells (starting with no memory influence?)
# - create model load interface 
# x get tensorboard visualization doing what we want
# 	x more: want to see weights / filters ideally... 	
# x create LIVE USER INPUT INTERFACE so we can do stuff like increase/decrease learning rate, save and quit, etc
				
################################### MAIN METHOD TESTING ################################
global LOAD_PREV_MODEL # If supplied, we will resume training a model from a previous checkpoint
LOAD_PREV_MODEL = ""
global LOAD_PREV_MODEL_STEPS
LOAD_PREV_MODEL_STEPS = 0
global FLAG_GENERATE_VIDEO
FLAG_GENERATE_VIDEO = 0
# TO LOAD A PREVIOUS MODEL, PLEASE SUPPLY THE .meta FILE
if __name__ == '__main__':
	print("Running program...")

	files = PrepareVideoFilesForTraining(VIDEO_NAME, VIDEO_FRAMES_FOLDER_NAME) # change constants at top of file

	VidFrames, NumFrames = CreateVidFrameTensor(files, WIDTH, HEIGHT)
	print("Number of frames: ",NumFrames)
	
	if len(sys.argv) > 1:
		LOAD_PREV_MODEL = str(sys.argv[1])
		LOAD_PREV_MODEL_STEPS = int(sys.argv[2])
		FLAG_GENERATE_VIDEO = int(sys.argv[3])
		print("LOADING PREVIOUS MODEL: ",LOAD_PREV_MODEL,"\n")
		if FLAG_GENERATE_VIDEO == 1:
			print("GENERATING TEST VIDEO FROM SUPPLIED MODEL...\n")
	else:
		print("LOAD_PREV_MODEL not given, starting new training session.\n")
		LOAD_PREV_MODEL = ""
		
	# In any case: TRAIN_FULL_MODEL will start a new model or load and continue training an existing one. 
	TRAIN_FULL_MODEL(VidFrames, NumFrames)
	
		
	
	print("\n================== END SCRIPT ====================\n")
