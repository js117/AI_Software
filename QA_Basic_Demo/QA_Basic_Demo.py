# Using Anaconda 3.5
# 
# V-REP instructions: https://www.youtube.com/watch?v=SQont-mTnfM 
# http://developers-club.com/posts/268313/ 

global ms1, ms2
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
import serial
import glob
import pygame
from pygame.locals import *
import threading

############################################## GLOBAL VARIABLES #########################################
global CAM_W
global CAM_H
global CAM_C
global NUM_CAM_CHANNELS
global vgg_weights_path

CAM_W = 224
CAM_H = 224
CAM_C = 3

def GetDataFromFolder(folder, do_normalize):
	
	curr_dir = os.getcwd()
	search_dir = ""
	if platform == "win32":
		search_dir = os.getcwd()+"\\"+folder # WINDOWS
	else:
		search_dir = os.getcwd()+"/"+folder # LINUX
	os.chdir(search_dir)
	files = filter(os.path.isfile, os.listdir(search_dir))
	files = [os.path.join(search_dir, f) for f in files] # add path to each file
	os.chdir(curr_dir)
	
	n = len(files)
	imgs = np.zeros((n, CAM_W, CAM_H, CAM_C), dtype=np.uint8)
	
	i = 0
	for f in files:
		img = cv2.resize(cv2.imread(f), (CAM_W, CAM_H))
		imgs[i,:,:,:] = img
		i = i + 1
		
	if (do_normalize == 1):	
		imgs = (imgs.astype(float) - 127.5) / 255 # normalize	
	return imgs
	
def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
	
def GetVGGModel():
	ms1 = time.time()*1000.0
	
	base_model = VGG16(weights='imagenet', include_top=False, input_shape=[CAM_W, CAM_H, 3])
	# Pick your output activation. See source:
	# https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py 
	model = Model(input=base_model.input, output=base_model.get_layer('block3_pool').output) # should be something like 32x32x128, since we started w/ 128x128 img
	
	ms2 = time.time()*1000.0
	print('VGG model loaded. Time elasped: ' +str(round((ms2-ms1)/1000, 2)) + ' secs')
	return model
	
def VGGFeatures(model, x): # x is image (n x W x H x 3), model is VGG
	ms1 = time.time()*1000.0
	features = model.predict(x)
	features = features / np.max(features) # normalize from 0 --> 1 for both our new network inputs, and for visualization
	ms2 = time.time()*1000.0
	print('VGG predict time: ' +str(round((ms2-ms1), 4)) + ' millis')
	return features
	
def FramesToVGGFeatures(VGG_model, frames): 
	vgg_cam_features = VGGFeatures(VGG_model, frames[:,:,:])

def normalizeImgForKeras(img):
	return np.expand_dims(((img.astype(float) - 127.5) / 255), axis=0)

global VGG_model	
global vgg_div_factor
global vgg_feat_maps
global CONV_1
global CONV_2
global CONV_3
global DENSE_1
global DENSE_2
global NUM_CLASSES

vgg_div_factor = 8	
vgg_feat_maps = 256
CONV_0 = 100
CONV_1 = 50
CONV_2 = 30
CONV_3 = 20
DENSE_1 = 150
DENSE_2 = 50
NUM_CLASSES = 3 # EMPTY, SUCCESS, DEFECT
	

	
def QAPredictorModel():

	# Actual images are 224 x 224 x 3. This model's input, using VGG, will be 28 x 28 x 256 (using block3_pool)
	#img_vgg = Input(shape=(round(CAM_W/vgg_div_factor), round(CAM_H/vgg_div_factor), vgg_feat_maps), name='input_img_vgg')
	
	#base_model = VGG16(weights='imagenet', include_top=False, input_shape=[CAM_W, CAM_H, 3])
	#network = base_model.get_layer('block1_pool').output
	img = Input(shape=(CAM_W, CAM_H, CAM_C), name='input_img_')
	network = Convolution2D(CONV_0, 3, 3, activation='elu', border_mode='same', name='conv_0')(img)	
	network = MaxPooling2D((2, 2), border_mode='same')(network) # out: 112 x 112 x 100
	network = Convolution2D(CONV_1, 3, 3, activation='elu', border_mode='same', name='conv_1')(network)	
	network = MaxPooling2D((2, 2), border_mode='same')(network) # out: 56 x 56 x 50
	network = Convolution2D(CONV_2, 3, 3, activation='elu', border_mode='same', name='conv_2')(network)
	network = MaxPooling2D((2, 2), border_mode='same')(network)	# out: 28 x 28 x 30
	network = Convolution2D(CONV_3, 3, 3, activation='elu', border_mode='same', name='conv_last')(network) # out: 7 x 7 x 32 ( == 1568 numbers)	
	network = MaxPooling2D((2, 2), border_mode='same', name='pool_last')(network) # out: 14 x 14 x 20 ( == 3920 #s)
	network = Flatten(name='flatten')(network)
	network = Dense(DENSE_1, activation='elu', name='dense_1')(network)
	#network = Dropout(0.5)(network)
	network = Dense(DENSE_2, activation='elu', name='dense_2')(network)
	#network = Dropout(0.5)(network)
	network = Dense(NUM_CLASSES, activation='softmax', name='dense_3')(network)
	
	model_inputs = img
	model_outputs = network 
	model = Model(input=model_inputs, output=model_outputs)
	
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	
	return model

############################################ HEATMAP ################################################
# https://github.com/jacobgil/keras-grad-cam
	
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes)) # tf.mul ? 

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
	
# image is preprocessed: normalize to 0-1, expand_dims	
def grad_cam(input_model, image, CAM_W, CAM_H, nb_classes, category_index, layer_name): 
    model = Sequential()
    model.add(input_model)

    #nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
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
	
############################################ HEATMAP ################################################	
		
class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape		

	
def raw_frame_to_vgg(x):
	x = (x.astype(float) - 127.5) / 255 
	x_vgg = VGGFeatures(VGG_model, np.expand_dims(x, axis=0))
	return x_vgg
	
def QAPM_Predict(x): # assumes x is just an UN-NORMALIZED frame from camera: (w,h,c)
	#x_ = (x.astype(float) - 127.5) / 255 
	#x_vgg = VGGFeatures(VGG_model, np.expand_dims(x_, axis=0))
	#pred = QAPM.predict(x_vgg)
	pred = QAPM.predict(x)
	return pred
	
def AugmentImageData(x, multiplier): # x is of shape (n, W, H, 3) and is normalized already
	n = x.shape[0]
	train_datagen = ImageDataGenerator(
										rotation_range=90,
										width_shift_range=0.1,
										height_shift_range=0.1,
										shear_range=0.1,
										zoom_range=0.4,
										horizontal_flip=True,
										vertical_flip=True,
										fill_mode='nearest')
	w, h, c = x.shape[1:]
	auged = np.zeros((round(n*multiplier), w, h, c))
	
	for i in range(n*multiplier):
		auged[i,:] = train_datagen.random_transform(x[randint(0,n-1),:])
		
	return np.concatenate((x, auged), axis=0) # returns original data + aug'd data
	
def GetTrainingData(in_memory_augment=0):

	# Hard-code testing for quicker development: 
	empty_dataset_folder = "empty3"
	success_dataset_folder = "success3"
	defect_dataset_folder = "defects3"

	# Use original as GT data
	train_empty = GetDataFromFolder(empty_dataset_folder, 1)			
	train_success = GetDataFromFolder(success_dataset_folder, 1)
	train_defects = GetDataFromFolder(defect_dataset_folder, 1)		# (n, W, H, 3) shape
	
	# TESTING:
	#train_empty = train_empty[0:10,:]
	#train_success = train_success[0:15,:]
	#train_defects = train_defects[0:20,:]
	
	n_empty = train_empty.shape[0]
	n_success = train_success.shape[0]
	n_defects = train_defects.shape[0]
	
	if (in_memory_augment == 1):
		train_empty = AugmentImageData(train_empty, 10) 
		train_success = AugmentImageData(train_success, 10)
		train_defects = AugmentImageData(train_defects, 10)
	
	#for i in range(0,10):
	#	cv2.imshow('frame',train_success_auged[100+i,:])
	#	cv2.waitKey(0)
	
	#train_empty_vgg = VGGFeatures(VGG_model, train_empty_auged)
	#train_success_vgg = VGGFeatures(VGG_model, train_success_auged)
	#train_defects_vgg = VGGFeatures(VGG_model, train_defects_auged)

	print(train_empty.shape)
	print(train_success.shape)
	print(train_defects.shape)	
	
	#n_empty_auged = train_empty_vgg.shape[0]
	#n_success_auged = train_success_vgg.shape[0]
	#n_defects_auged = train_defects_vgg.shape[0]
	n_empty = train_empty.shape[0]
	n_success = train_success.shape[0]
	n_defects = train_defects.shape[0]
	# ORDERS MUST CORRESPOND! 
	X_train = np.concatenate((train_empty, train_success, train_defects), axis=0)
	train_labels = np.array([0] * n_empty + [1] * n_success + [2] * n_defects)
	Y_train = to_categorical(train_labels, 3)
	
	X_val = np.concatenate((train_empty[0:n_empty,:], train_success[0:n_success,:], train_defects[0:n_defects,:]), axis=0)
	val_labels = np.array([0] * n_empty + [1] * n_success + [2] * n_defects)
	Y_val = to_categorical(val_labels, 3)

	return X_train, Y_train, X_val, Y_val
	
def TrainOnData(in_memory_augment=0, nb_epoch=30):
	
	X_train, Y_train, X_val, Y_val = GetTrainingData(in_memory_augment=0)
	
	print(X_train.shape)
	print(Y_train.shape)
	print(X_val.shape)
	print(Y_val.shape)
	
	print(X_train.nbytes / (2**20))
	
	tb = TensorBoard(log_dir='logs')
	checkpointer = ModelCheckpoint(filepath="QAPM_test_curr_best.h5", verbose=1, save_best_only=True)
	if (in_memory_augment == 1):
		QAPM.fit(X_train, Y_train,
						nb_epoch=nb_epoch,
						batch_size=20,
						shuffle=True,
						validation_data=(X_val, Y_val),
						callbacks=[tb, checkpointer])
	else:
		train_datagen = ImageDataGenerator(
										rotation_range=30,
										width_shift_range=0.1,
										height_shift_range=0.1,
										#shear_range=0.1,
										#zoom_range=0.5,
										horizontal_flip=True,
										vertical_flip=True,
										fill_mode='nearest')
		QAPM.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=10),
											  samples_per_epoch=len(X_train), 
											  validation_data=(X_val, Y_val),
											  nb_epoch=nb_epoch,
											  callbacks=[tb, checkpointer])			
					
	QAPM_str_name = 'QAPM_test'+str(time.time()).replace(".","")+'.h5' # pro-tip: manually re-name after each run... 
	QAPM.save(QAPM_str_name)
	
def InitModel(weights_str):
	QAPM.load_weights(weights_str)
	
	QAPM.summary()
	
	#HEATMAP_MODEL = classificationModelToHeatmap(QAPM, QAPM_Heatmap())
	
def ScreenUpdateThread(): # call after appropriate structures have been initialized
	#pygame.fastevent.init() 
	while True:
		ret, frame = cap.read()	
		f = frame[0:cam_height,x1:x2,:]
		
		rows,cols = (f.shape[0], f.shape[1])
		M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
		f = cv2.warpAffine(f,M,(cols,rows))

		
		
		if (is_analyzing == 1):
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(f,'Analyzing...',(20,20), font, 0.5,(255,0,0),1,cv2.LINE_AA)
		
		if (is_realtime_analysis == 1 and is_analyzing == 0): # other thread will time how long this happens for
			ps = {}; ps['P_empty'] = p_empty; ps['P_success'] = p_success; ps['P_defect'] = p_defect
			strr = ""
			for w in sorted(ps, key=ps.get, reverse=True):
				strr = strr+str(w)+" / "+str(ps[w])+"\n"
				#print(w, ps[w])
			font = cv2.FONT_HERSHEY_SIMPLEX
			y0, dy = 20, 20
			for i, line in enumerate(strr.split('\n')):
				y = y0 + i*dy
				cv2.putText(f, line, (20, y ), font, 0.5,(0,0,255),1,cv2.LINE_AA)
			#cv2.putText(f,strr,(20,20), font, 0.5,(0,0,255),1,cv2.LINE_AA)

		if (is_recording == 1):
			timestamp = str(time.time()).replace(".","")
			img_name = timestamp+'.jpg'
			cv2.imwrite(recording_folder+"/"+img_name, f)	
			
		ff = pygame.surfarray.make_surface(f)
		ff = pygame.transform.rotate(ff, 90)
		ff = pygame.transform.flip(ff, False, True)
		screen.blit(ff,(0,0))
		pygame.display.flip()
		
		
		
		#if (is_realtime_analysis == 1): # use global variables to update UI
				
				
def PrintTopLevelMenu():
	print("\n ---------- Instructions ----------\n")
	print("a: Create New Dataset: empty scene (no products)")
	print("s: Create New Dataset: successful products")
	print("d: Create New Dataset: defective products")
	print("t: Train New Model (will ask user to specify folder names)")
	print("r: Run real-time detection (will ask user to specify model file by name)")
	print("z: Capture image for dataset (only when creating new dataset)")
	print("c: Configure robot serial port control")
	print("p: Enter (and exit) robot positioning mode")
	print("q: Quit.")
	print("---------------------------------------")
	
global cam_source
global model_width	
global model_height
global cam_width	
global cam_height
global cap
global screen
global screen_thread
global x1							# used for cropping to square
global x2							# used for cropping to square
global is_creating_new_dataset		# userflow flag
global new_dataset_folder			
global empty_dataset_folder
global success_dataset_folder
global defect_dataset_folder
global is_robot_setup				# userflow flag
global is_setting_up_robot			# userflow flag
global is_positioning_mode			# userflow flag
global is_realtime_analysis			# userflow flag
global is_analyzing
global is_recording
global recording_folder
global p_empty, p_success, p_defect
global VGG_model
global QAPM
global HEATMAP_MODEL
# Robot-specific params
global ports
global baudrate
global num_speed_dx_cmds
global num_steps_dx_cmds
global MOTOR_shoulder_yaw 
global MOTOR_shoulder_pitch 
global MOTOR_elbow_pitch 
global MOTOR_wrist_pitch 
global q,w,e,r,t,y
	
if __name__ == '__main__':
	print("Running program...")
	
	QAPM = QAPredictorModel()
	
	InitModel("QAPM_aws_best.h5")
	# or
	#TrainOnData(); sys.exit()
	
	########## Test out heatmap stuff: #########
	'''X_train, Y_train, _, _ = GetTrainingData()
	print(X_train.shape); print(Y_train.shape)
	
	for image in X_train[30:40,:]:
		img = np.expand_dims(image, axis=0)
		print(img.shape)
		heatmap_img = grad_cam(QAPM, img, CAM_W, CAM_H, nb_classes=3, category_index=0, layer_name="pool_last")
		cv2.imshow('asdf', np.concatenate((image,heatmap_img), axis=0))
		cv2.waitKey(0)
	
	sys.exit()'''
	
	
	print("\n========== Welcome to the Big Solve Robotics QA Basic Demo ==========\n")
	cam_source = int(sys.argv[1])	
	cap = cv2.VideoCapture(cam_source) # ==1 for USB webcam
	pygame.init() 					   # initialize pygame
	ret, frame = cap.read()
	cam_width = frame.shape[1]
	cam_height = frame.shape[0]
	screen = pygame.display.set_mode((cam_height, cam_height)) # this square corresponds to size of img we capture
	x1 = round((cam_width-cam_height)/2); x2 = x1 + cam_height
	is_creating_new_dataset = 0
	new_dataset_folder = ""
	empty_dataset_folder = ""
	success_dataset_folder = ""
	defect_dataset_folder = ""
	is_robot_setup = 0
	is_setting_up_robot = 0
	is_positioning_mode = 0
	is_realtime_analysis = 0
	is_analyzing = 0
	is_recording = 0
	p_empty = 0; p_success = 0; p_defect = 0
	ports = []
	baudrate = 19200
	num_speed_dx_cmds = 10
	num_steps_dx_cmds = 10
	MOTOR_shoulder_yaw = 0 
	MOTOR_shoulder_pitch = 0 
	MOTOR_elbow_pitch = 0 
	MOTOR_wrist_pitch = 0 
	q = "q"; q = q.encode('utf-8')
	w = "w"; w = w.encode('utf-8')
	t = "t"; t = t.encode('utf-8')
	y = "y"; y = y.encode('utf-8')
	e = "e"; e = e.encode('utf-8')
	r = "r"; r = r.encode('utf-8')
	
	# new_dataset_empty = input("Enter your dataset name, single word:\n\n")
	PrintTopLevelMenu()
	
	screen_thread = threading.Thread(target=ScreenUpdateThread)
	try:
		screen_thread.setDaemon(True)  # important for cleanup ? 
		screen_thread.start() # join too? 
	except (KeyboardInterrupt, SystemExit):
		cleanup_stop_thread();
		sys.exit()
	
	while(True):
		#k = cv2.waitKey(1) & 0xFF
		pygame.event.pump()
		#pygame.event.poll()
		#pygame.event.wait()
		keys = pygame.key.get_pressed()  #checking pressed keys
		
		#if (is_realtime_analysis == 1):
			#print("do stuff")
	
		# KEY PRESS FUNCTIONALITY
		if (is_positioning_mode == 0):
		
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
		
					if event.key == K_z: #k == ord('z'): # save screenshot; this interacts w/ a,s,d key settings
						if (is_creating_new_dataset == 1):
							timestamp = str(time.time()).replace(".","")
							img_name = timestamp+'.jpg'
							print("\nSaving image: "+img_name)
							ret, frame = cap.read()
							f = frame[0:cam_height,x1:x2,:]
							cv2.imwrite(new_dataset_folder+"/"+img_name, f)
							continue
						
						
					if event.key == K_a: #k == ord('a'):
						if (is_creating_new_dataset == 0):
							print("\nCreating New Dataset: empty scene (no products)\n")
							print("Instructions: press 'z' to save images relevant to your dataset.")
							print("Recommended methodology: at least 20 images, of various scales and rotations.")
							print("When finished, press 'a' again to exit.")
							timestamp = str(time.time()).replace(".",""); timestamp = timestamp[-5:]
							new_dataset_folder = str(input("Name the folder of your new dataset: "))
							os.makedirs(new_dataset_folder)
							print("Your folder name: " + new_dataset_folder)
							is_creating_new_dataset = 1
							continue
						if (is_creating_new_dataset == 1):
							print("\nFinished creating dataset.\n")
							is_creating_new_dataset = 0
							new_dataset_folder = ""
							continue
					
					if event.key == K_s: #k == ord('s'):
						if (is_creating_new_dataset == 0):
							print("\nCreating New Dataset: successful products\n")
							print("Instructions: press 'z' to save images relevant to your dataset.")
							print("Recommended methodology: at least 20 images, of various scales and rotations.")
							print("When finished, press 's' again to exit.")
							timestamp = str(time.time()).replace(".",""); timestamp = timestamp[-5:]
							new_dataset_folder = str(input("Name the folder of your new dataset: "))
							os.makedirs(new_dataset_folder)
							print("Your folder name: " + new_dataset_folder)
							is_creating_new_dataset = 1
							continue
						if (is_creating_new_dataset == 1):
							print("\nFinished creating dataset.\n")
							is_creating_new_dataset = 0
							new_dataset_folder = ""
							continue
					
					if event.key == K_d: #k == ord('d'):
						if (is_creating_new_dataset == 0):
							print("\nCreating New Dataset: defective products\n")
							print("Instructions: press 'z' to save images relevant to your dataset.")
							print("Recommended methodology: at least 20 images, of various scales and rotations.")
							print("When finished, press 'd' again to exit.")
							timestamp = str(time.time()).replace(".",""); timestamp = timestamp[-5:]
							new_dataset_folder = str(input("Name the folder of your new dataset: "))
							os.makedirs(new_dataset_folder)
							print("Your folder name: " + new_dataset_folder)
							is_creating_new_dataset = 1
							continue
						if (is_creating_new_dataset == 1):
							print("\nFinished creating dataset.\n")
							is_creating_new_dataset = 0
							new_dataset_folder = ""
							continue
						
					if event.key == K_t: #k == ord('t'):
						print("\n ::: Training Mode ::: \n")
						empty_dataset_folder = input("Enter the name of the 'empty scene' dataset folder: ")
						success_dataset_folder = input("Enter the name of the 'successful products' dataset folder: ")
						defect_dataset_folder = input("Enter the name of the 'defective products' dataset folder: ")
						print("\nCamera paused, running vision training program...\n")
						TrainOnData()
						continue
						
					if event.key == K_c: #k == ord('c'):
						if (is_robot_setup == 0):
							print("Configure available ports by testing motors one at a time.")
							print("Assign each port to a motor, press the following if the current port corresponds with the motor: \n")
							print("  a -- base rotation\n s -- shoulder pitch\n d -- elbow\n f -- wrist\n")
							print("(Press 'q' and 'w' to test a motor)\n")
							ports = [str(p) for p in serial_ports()]
							print(ports)
							is_setting_up_robot = 1
							for port in ports:
								print("Current port: "+port)
								MOTOR_test = serial.Serial(port, baudrate)
								while True:
									if msvcrt.kbhit():
										n = msvcrt.getch().decode("utf-8").lower()
										if (n == 'q'):
											MOTOR_test.write(q);
										if (n == 'w'):
											MOTOR_test.write(w);
										if (n == 'a'):
											MOTOR_test.close() 
											MOTOR_shoulder_yaw = serial.Serial(port, baudrate) 
											print("Port "+port+" selected as base rotation joint\n"); break
										if (n == 's'):
											MOTOR_test.close() 
											MOTOR_shoulder_pitch = serial.Serial(port, baudrate) 
											print("Port "+port+" selected as shoulder pitch joint\n"); break
										if (n == 'd'):
											MOTOR_test.close() 
											MOTOR_elbow_pitch = serial.Serial(port, baudrate) 
											print("Port "+port+" selected as elbow joint\n"); break
										if (n == 'f'):
											MOTOR_test.close() 
											MOTOR_wrist_pitch = serial.Serial(port, baudrate) 
											print("Port "+port+" selected as wrist joint\n"); break
							print("\nFinished configuring ports.\n")				
							is_robot_setup = 1	
							continue
											
					if event.key == K_r: #k == ord('r'):
						print("\n --- REAL-TIME ANALYSIS ---\n")
						if (is_realtime_analysis == 0):
							is_realtime_analysis = 1	# and then it's just on.. 
						is_analyzing = 1
						
						ret, frame = cap.read()
						f = frame[0:cam_height,x1:x2,:]
						img = cv2.resize(f, (CAM_W, CAM_H))
						pred = QAPM_Predict(normalizeImgForKeras(img))
						print(pred[0])
						p_empty = round(pred[0][0], 3) * 100
						p_success = round(pred[0][1], 3) * 100
						p_defect = round(pred[0][2], 3) * 100
						
						is_analyzing = 0
						ms1 = time.time()*1000.0
						while (time.time()*1000.0 - ms1 < 5000): # for a few secs display analysis results
							pass 
						is_realtime_analysis = 0
						
						continue
						
					if event.key == K_p: #k == ord('p'):
						if (is_robot_setup == 1):
							print("\nENTERING ROBOT POSITIONING MODE\n")
							is_positioning_mode = 1
						else:
							print("Robot COM ports not setup. Please press 'c' from main menu to configure.\n")
						continue
						
					if event.key == K_v: #k == ord('v'):
						if (is_recording == 0):
							recording_folder = str(time.time()).replace(".",""); recording_folder = "rec_"+recording_folder[-5:]
							print("\nNew recording: "+recording_folder+"\n")
							os.makedirs(recording_folder)
							is_recording = 1
						elif (is_recording == 1):
							is_recording = 0
						
					if event.key == K_q: #k == ord('q'):
						print("\nGoodbye.\n")
						screen_thread.join(0)
						sys.exit()
			
		if (is_positioning_mode == 1): 
				
			# TODO: organize this code somehow... e.g. custom funcs for custom speed & step changes	
			############## Motor actuation: ##############
			if keys[pygame.K_1]: #k == ord('1'):
				MOTOR_shoulder_yaw.write(q); time.sleep(.001)
			if keys[pygame.K_2]: #k == ord('2'):
				MOTOR_shoulder_yaw.write(w); time.sleep(.001)
			if keys[pygame.K_q]: #k == ord('q'):
				MOTOR_shoulder_pitch.write(q); time.sleep(.001)
			if keys[pygame.K_w]: #k == ord('w'):
				MOTOR_shoulder_pitch.write(w); time.sleep(.001)
			if keys[pygame.K_a]: #k == ord('a'):
				MOTOR_elbow_pitch.write(q); time.sleep(.001)
			if keys[pygame.K_s]: #k == ord('s'):
				MOTOR_elbow_pitch.write(w); time.sleep(.001)
			if keys[pygame.K_z]: #k == ord('z'):
				MOTOR_wrist_pitch.write(q); time.sleep(.001)
			if keys[pygame.K_x]: #k == ord('x'):
				MOTOR_wrist_pitch.write(w); time.sleep(.001)
			
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
				
					if event.key == K_p: #k == ord('p'):
						print("\n -- Returning to top-level menu -- \n")
						is_positioning_mode = 0
						PrintTopLevelMenu()
						continue
				
					# check all the key event and update status
					# Control speed:                 
					if event.key == K_3:
						MOTOR_shoulder_yaw.write(t)
					if event.key == K_4:  
						MOTOR_shoulder_yaw.write(y)
						
					if event.key == K_e:
						MOTOR_shoulder_pitch.write(t)
					if event.key == K_r:  
						MOTOR_shoulder_pitch.write(y)
						
					if event.key == K_d:
						MOTOR_elbow_pitch.write(t)
					if event.key == K_f:  
						MOTOR_elbow_pitch.write(y)
						
					if event.key == K_c:
						MOTOR_wrist_pitch.write(t)
					if event.key == K_v:  
						MOTOR_wrist_pitch.write(y)
						
					# Control step size:                 
					if event.key == K_5:
						MOTOR_shoulder_yaw.write(e)
					if event.key == K_6:  
						MOTOR_shoulder_yaw.write(r)
						
					if event.key == K_t:
						MOTOR_shoulder_pitch.write(e)
					if event.key == K_y:  
						MOTOR_shoulder_pitch.write(r)
						
					if event.key == K_g:
						MOTOR_elbow_pitch.write(e)
					if event.key == K_h:  
						MOTOR_elbow_pitch.write(r)
						
					if event.key == K_b:
						MOTOR_wrist_pitch.write(e)
					if event.key == K_n:  
						MOTOR_wrist_pitch.write(r)
				
	