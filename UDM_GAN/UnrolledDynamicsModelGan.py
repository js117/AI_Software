
# coding: utf-8

# In[70]:

import sys
import os 
import random
from random import randint
import numpy as np
import cv2
sys.path.append('..')
import Video_Utils
import CNN_Utils
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

'''
From the command line you can convert a notebook to python with this command:

ipython nbconvert --to python <YourNotebook>.ipynb

You may have to install the python mistune package:

sudo pip install mistune
'''


# In[71]:

############################################## GLOBAL VARIABLES #########################################
# These variables are specific to the dataset of the problem. 
# Our data is the following: 
#
# time-series of 4 cameras (128 x 128 x 3) looking at a robot complete a task (pick up object, place in bin)
# + corresponding time series of 7-d vector of joint commands (6 DOF + gripper) denoting the position delta command.
#
# In future comments, we will refer to the states and actions as x(t) and u(t) respectively, where 'state' 
# denotes the visual data sensed from the world (the 4 cameras) and action is the 7-d joint position delta vector. 
#
# Data was generated and collected using the V-REP (http://www.coppeliarobotics.com/) robot simulator 
# (free/education version). Please contact the author of this notebook for the specific scene and data-generation
# script 

global CAM_W
global CAM_H
global CAM_C
global NUM_CAMS
global ACTION_LEN

CAM_W = 128
CAM_H = 128
CAM_C = 3
NUM_CAMS = 4
ACTION_LEN = 7


# In[72]:

####################################### MODEL PARAMETERS #########################################
global NUM_FUTURE_FRAMES
global NUM_PAST_FRAMES
#global NUM_VGG_FEAT_MAPS
#global VGG_FEAT_W
#global VGG_FEAT_H 
global CONV1_FEAT_MAPS
global CONV2_FEAT_MAPS
global CONV3_FEAT_MAPS
global CONVP1_FEAT_MAPS 
global CONVP2_FEAT_MAPS
global CONVP3_FEAT_MAPS
global CONVF1_FEAT_MAPS
global CONVF2_FEAT_MAPS
global CONVF3_FEAT_MAPS
global CONVD1_FEAT_MAPS
global CONVD2_FEAT_MAPS
global CONVD3_FEAT_MAPS
global D_DENSE1
global D_DENSE2
global D_DENSE3
global UDM_model       # The actual Keras model

#VGG_FEAT_W = 64        # using "block1_pool"
#VGG_FEAT_H = 64        # using "block1_pool"  
#NUM_VGG_FEAT_MAPS = 64 # using "block1_pool" (per camera)
NUM_FUTURE_FRAMES = 10
NUM_PAST_FRAMES = 10
# Below: these layers create a more data-efficient representation of past from preprocessed features
CONVP1_FEAT_MAPS = round(CAM_C*NUM_CAMS*NUM_PAST_FRAMES/2)        # e.g. 120/2 = 60
CONVP2_FEAT_MAPS = round(CONVP1_FEAT_MAPS/2)                      # e.g. 30
CONVP3_FEAT_MAPS = round(CONVP2_FEAT_MAPS/2)                      # e.g. 15
# Below: these layers create a more data-efficient representation from merged frame,action [from the past] inputs
CONV1_FEAT_MAPS = CONVP3_FEAT_MAPS
CONV2_FEAT_MAPS = CONVP3_FEAT_MAPS
CONV3_FEAT_MAPS = NUM_PAST_FRAMES
# Below: these are final output layers that transform merged input data to predicted output frames
CONVF1_FEAT_MAPS = (CONV3_FEAT_MAPS + 1)*2                        # e.g. 22
CONVF2_FEAT_MAPS = round(CONVF1_FEAT_MAPS*2)                      # e.g. 48
CONVF3_FEAT_MAPS = round(CONVF2_FEAT_MAPS*2)                      # e.g. 96
CONVF4_FEAT_MAPS = round(CAM_C*NUM_CAMS*NUM_PAST_FRAMES)  # e.g. 3*4*NUM_PAST_FRAMES = 120 
# Below: parameters to learn important features in the sequence of future actions 
D_DENSE3 = CAM_W * CAM_H
D_DENSE2 = round(D_DENSE3 / 8)
D_DENSE1 = round(D_DENSE2 / 8)


# In[77]:

################################################# THE MODEL ####################################################
'''
An explanation of the dynamics model and the buffers: 

We are training a model to produce p(x(t+1: t+F) | x(t-P: t), u(t-P: t), u(t+1: t+F-1)) 

I.e.: predict the future F frames, 
      given the past P frames, past P actions, and the future F-1 actions 
      
This can be viewed as unrolling the classic p(x(t+1) | x(t), u(t)) dynamics model. 

We have proposed this 'unrolled dynamics' problem formulation for the following reasons:

1) Conditioning on longer sequences to capture the effects of repeated actions, reduce discretization errors
2) Improved training due to longer sequences. Single-step visual models have a tendency to "copy" prev. frame
   as the output
3) Be able to visualize a future state trajectory based on a proposed set of actions.
   ---> this has applications in control planning, and can be useful to a human operator/co-worker, or for
        another system to check for safety hazards, etc. Plus makes a sick demo ("this is what the robot is
        thinking") lol. 
        
PREV_FRAMES_BUFFER ---> input, this is x(t-P: t)         (size P,W,H,3*num_cams)
PREV_ACTION_BUFFER ---> input, this is u(t-P: t)         (size P,A)
FUTURE_FRAMES_BUFFER ---> output, this is x(t+1: t+F)    (size F,W,H,3*num_cams)
FUTURE_ACTION_BUFFER ---> input, this is u(t+1: t+F-1)   (size F-1,A)           [if this was output, we'd be doing control]

(note: visual inputs may be pre-processed with another pre-trained model)

Other details/throughts: 
- ELU activations: to capture training efficiency of ReLU while reducing need for batch normalization
- multi-frame unrolling vs. single frame p(x(t+1)|x(t),u(t)): unrolling is like doing batches of the single-frame
  version, except the model can correlated (x,y)_t points in the batch   
- The size of our conv layers (measured by # of filters per layer) creates a "bottleneck": form a more data-efficient
  representation of the past information, combine with future action information, and then generate the larger tensor
  of future predicted frames. 

'''
# Can use part of pre-trained VGG model to seed features with reasonable features: (used online during training)  
#vgg_preprocessor = CNN_Utils.GetVGGModel("block1_pool", CAM_W, CAM_H, print_timing=1) 
# ^ Note: not using for this experiment (creates too many feature maps per camera for current hardware)

# Model input #1: past frames
input_prev_frames_raw = Input(shape=(CAM_W, CAM_H, CAM_C*NUM_CAMS*NUM_PAST_FRAMES), name='input_prev_frames_raw')
# below: some layers to learn what information is important from the past
input_prev_frames = Convolution2D(CONVP1_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames_raw)
input_prev_frames = Convolution2D(CONVP2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames)
input_prev_frames = Convolution2D(CONVP3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same')(input_prev_frames) 

# Model input #1: past actions 
input_prev_actions_raw = Input(shape=(NUM_PAST_FRAMES*ACTION_LEN,), name='input_prev_actions_raw')
# Below: project and reshape, so we can merge inputs later: 
input_prev_actions_p = Dense(output_dim=CAM_W*CAM_H*1, activation='elu')(input_prev_actions_raw)
input_prev_actions_p_r = Reshape((CAM_W, CAM_H, 1), name='input_prev_actions_p_r')(input_prev_actions_p)

# Model input #3: future actions
input_future_actions_raw = Input(shape=((NUM_FUTURE_FRAMES-1)*ACTION_LEN,), name='input_future_actions_raw')
# format for convenience, to let us 'pick off' actions and sequentially predict the next frame logically
# Below: these parameters gather information for an action pertaining to how it affects a future state
D_A1 = Dense(output_dim=D_DENSE1, activation='elu', name='D_A1')(input_future_actions_raw)
D_A2 = Dense(output_dim=D_DENSE2, activation='elu', name='D_A2')(D_A1)
D_A3 = Dense(output_dim=D_DENSE3, activation='elu', name='D_A3')(D_A2)
future_action_branch = Reshape((CAM_W, CAM_H, 1), name='action_branch_r')(D_A3)

# MERGE PAST INPUTS: 
merged_input = merge([input_prev_frames, input_prev_actions_p_r], 
                      mode='concat', concat_axis=3, name='merged_prev_inputs')
# ^ so now past frames, info from past actions has been merged into a tensor that is size e.g.:
# [W, H, CAM_C*NUM_CAMS*NUM_PAST_FRAMES/8 + 1] (e.g. == 16 for 10 past frames)

# Pre-process merged inputs that contain past information:
# This is meant to form a more efficient representation to be used in predicting future frames given proposed actions:
merged_input = Convolution2D(CONV1_FEAT_MAPS, 7, 7, activation='elu', border_mode='same', name='merged_lvl_1')(merged_input)
merged_input = Convolution2D(CONV2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='merged_lvl_2')(merged_input)
merged_input = Convolution2D(CONV3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='merged_lvl_3')(merged_input)

# Now merge with information about future actions: 
merged_input = merge([merged_input, future_action_branch], 
                      mode='concat', concat_axis=3, name='merged_efficient_inputs_all')
# ^ This will have size [W, H, NUM_PAST_FRAMES + 1], e.g. [128, 128, 11]

# Conv layers for predicting the next frames given all relevant data: 
Out_C1 = Convolution2D(CONVF1_FEAT_MAPS, 7, 7, activation='elu', border_mode='same', name='Out_C1')(merged_input)
Out_C2 = Convolution2D(CONVF2_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='Out_C2')(Out_C1)
Out_C3 = Convolution2D(CONVF3_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='Out_C3')(Out_C2)
Out_C4 = Convolution2D(CONVF4_FEAT_MAPS, 3, 3, activation='elu', border_mode='same', name='Out_C4')(Out_C3)
# ^ Out_C4 is a final output layer, spits out a frame (defined as WxHx3*num_cams*num_future_frames)

# Define the full model structure: 
model_inputs = [input_prev_frames_raw, input_prev_actions_raw, input_future_actions_raw]
# Review of input dimensions: (caller of train/predict must expand_dims to achieve shapes)
# input_prev_frames_raw: [W, H, 3*NUM_CAMS*NUM_PAST_FRAMES] e.g. [1, 128, 128, 120]
# input_prev_actions_raw: [NUM_PAST_FRAMES*ACTION_LEN,] e.g. [1, 70] 
# input_future_actions_raw: [(NUM_FUTURE_FRAMES-1)*ACTION_LEN,] e.g. [1, 63]
model_outputs = [Out_C4] 
# Target output data: 
# Out_C4: [W, H, 3*NUM_CAMS*NUM_FUTURE_FRAMES], e.g. [1, 128, 128, 120]
UDM_model = Model(input=model_inputs, output=model_outputs)

UDM_optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
UDM_model.compile(loss='binary_crossentropy', optimizer=UDM_optimizer) 
# ^ TODO: custom loss function more appropriate for sequence of similar images...   

UDM_model.summary()


# In[78]:

####################################### INITIALIZATION, SETUP OF MODEL #########################################


# In[79]:

########################################## Globals for training params ####################################

global TOTAL_TRAINING_ITRS
global SAVE_CHECKPOINT_ITRS
global NUM_DEMONSTRATIONS
global CURR_DEMONSTRATION
global LENGTH_CURR_DEMONSTRATION # e.g. current demo we're looking at is 230 timesteps
global T_CURR_DEMONSTRATION      # and e.g. we're currently on timestep 87
global PERCENT_TRAIN             # percent of data used for training vs. valudation
global DEMONSTRATION_FOLDERS
global TRAINING_FOLDERS
global TESTING_FOLDERS
global NUM_TRAINING_DEMONSTRATIONS
global NUM_TESTING_DEMONSTRATIONS
global IMAGE_COMPARE_CHECKPOINT

global PREV_FRAMES_BUFFER
global PREV_ACTION_BUFFER
global FUTURE_FRAMES_BUFFER
global FUTURE_ACTION_BUFFER
global MODEL_LOSS_BUFFER

PREV_FRAMES_BUFFER = np.zeros((NUM_PAST_FRAMES, CAM_W, CAM_H, CAM_C))
PREV_ACTION_BUFFER = np.zeros((NUM_PAST_FRAMES, ACTION_LEN))
FUTURE_FRAMES_BUFFER = np.zeros((NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C))
FUTURE_ACTION_BUFFER = np.zeros((NUM_FUTURE_FRAMES-1, ACTION_LEN))

TOTAL_TRAINING_ITRS = 100000
SAVE_CHECKPOINT_ITRS = 100
IMAGE_COMPARE_CHECKPOINT = 5
CURR_DEMONSTRATION = -1 # start on the first one, loop to another one each itr
LENGTH_CURR_DEMONSTRATION = -1
PERCENT_TRAIN = 0.75
MODEL_LOSS_BUFFER = np.zeros((10,1))


# In[ ]:

########################################## START THE TRAINING #######################################

print("Running program...")

DEMONSTRATION_FOLDERS = Video_Utils.GetFoldersForRuns()

NUM_DEMONSTRATIONS = len(DEMONSTRATION_FOLDERS)

for f in DEMONSTRATION_FOLDERS:
    print(f)

# Separate into training and testing data based on the number of recordings (folders) we have: 
random.shuffle(DEMONSTRATION_FOLDERS)    
lim_separate = round(NUM_DEMONSTRATIONS * PERCENT_TRAIN)    
TRAINING_FOLDERS = DEMONSTRATION_FOLDERS[0:lim_separate]
TESTING_FOLDERS = DEMONSTRATION_FOLDERS[lim_separate:]
NUM_TRAINING_DEMONSTRATIONS = len(TRAINING_FOLDERS)
NUM_TESTING_DEMONSTRATIONS = len(TESTING_FOLDERS)

print(NUM_TRAINING_DEMONSTRATIONS); print(NUM_TESTING_DEMONSTRATIONS); print(NUM_DEMONSTRATIONS) # a + b = c

    

# Data flow process: we train on entire folder (sample run of a robot) before moving on to the next to 
# amortize the time required to load that folder's training data into RAM (multiple seconds). For a dynamics
# model this should be perfectly acceptable because the dynamics to be learned are ideally the *same* between
# different sample runs recorded on the (simulated) robot. 
itrs = 0
while itrs < TOTAL_TRAINING_ITRS:
    
    # Choose which expert demonstration we're using: 
    CURR_DEMONSTRATION = randint(0,NUM_TRAINING_DEMONSTRATIONS-1)

    frames, actions = Video_Utils.LoadFramesActionsFromFolder(TRAINING_FOLDERS[CURR_DEMONSTRATION], CAM_W, CAM_H, CAM_C*NUM_CAMS, ACTION_LEN)
    # ^ about 5 secs
    
    print(frames.shape)
    print(actions.shape)
    
    LENGTH_CURR_DEMONSTRATION = frames.shape[0] # number of timesteps for this demonstration
    
    # New demonstration, reset the relevant data buffers:
    PREV_FRAMES_BUFFER = np.zeros((NUM_PAST_FRAMES, CAM_W, CAM_H, CAM_C*NUM_CAMS))
    PREV_ACTION_BUFFER = np.zeros((NUM_PAST_FRAMES, ACTION_LEN))
    FUTURE_FRAMES_BUFFER = np.zeros((NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C*NUM_CAMS))
    FUTURE_ACTION_BUFFER = np.zeros((NUM_FUTURE_FRAMES-1, ACTION_LEN))
    
    print("\n===== Training on robot sample run #:"+str(CURR_DEMONSTRATION)+" with num timesteps: "+str(LENGTH_CURR_DEMONSTRATION)+"\n")
    
    for i in range(0, round(LENGTH_CURR_DEMONSTRATION/NUM_FUTURE_FRAMES)): 
        # only sample a limited number from each demo before moving on to next demo. 
        t = randint(0,LENGTH_CURR_DEMONSTRATION-1)	# now we have random shuffle training within a demo.
            
        ##### STEP 1: Load Buffers with Data #####    
        ms1 = time.time()*1000.0 
        # Past data: (up to and including x(t), and u(t) - which causes x(t+1), etc)
        if (t >= NUM_PAST_FRAMES): # regular, in-bounds
            PREV_FRAMES_BUFFER = frames[t-NUM_PAST_FRAMES:t, :] 
            PREV_ACTION_BUFFER_BUFFER = actions[t-NUM_PAST_FRAMES:t, :]
        else: # t is less than past frames, need to concat zeros at beginning
            LIM1 = np.abs(t - NUM_PAST_FRAMES) 
            PREV_FRAMES_BUFFER = np.concatenate((np.zeros((LIM1, CAM_W, CAM_H, CAM_C*NUM_CAMS)), frames[0:t, :]), axis=0) 
            PREV_ACTIONS_BUFFER = np.concatenate((np.zeros((LIM1, ACTION_LEN)), actions[0:t, :]), axis=0)
        # Future Data: (include 1 less future action than frame to retain logical p(x(t+1)|x(t),u(t)) structure)
        if (t <= LENGTH_CURR_DEMONSTRATION - NUM_FUTURE_FRAMES - 1):
            FUTURE_FRAMES_BUFFER = frames[t:t+NUM_FUTURE_FRAMES, :]
            FUTURE_ACTIONS_BUFFER = actions[t:t+NUM_FUTURE_FRAMES-1, :]
        else: # running off the end
            LIM2 = t + NUM_FUTURE_FRAMES - LENGTH_CURR_DEMONSTRATION 
            # ^ We need to repeat last frame this many times / add this many zero actions
            FUTURE_FRAMES_BUFFER = frames[t:LENGTH_CURR_DEMONSTRATION, :]
            last_frame = np.expand_dims(frames[LENGTH_CURR_DEMONSTRATION-1], axis=0)
            for j in range(0,LIM2):
                FUTURE_FRAMES_BUFFER = np.concatenate((FUTURE_FRAMES_BUFFER, last_frame), axis=0) 
            FUTURE_ACTIONS_BUFFER = np.concatenate((actions[t:LENGTH_CURR_DEMONSTRATION, :], np.zeros((LIM2, ACTION_LEN))), axis=0)
        ms2 = time.time()*1000.0
        print("\nLoaded data for timestep "+str(t)+" in "+str(round((ms2-ms1)/1, 3))+" msecs\n")
        
        #print(PREV_FRAMES_BUFFER.shape)
        #print(PREV_ACTION_BUFFER.shape)
        #print(FUTURE_FRAMES_BUFFER.shape)
        #print(FUTURE_ACTION_BUFFER.shape)
        
        ##### STEP 2: Train on Batch of Data Gathered Above #####
        ms1 = time.time()*1000.0
        # inputs: [input_prev_frames_raw, input_prev_actions_raw, input_future_actions_raw]
        # outputs/training targets: [future_frames]
        model_loss = UDM_model.train_on_batch(
                     [np.expand_dims(PREV_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_CAMS*NUM_PAST_FRAMES), axis=0), 
                      np.expand_dims(PREV_ACTION_BUFFER.reshape(-1), axis=0), 
                      np.expand_dims(FUTURE_ACTION_BUFFER.reshape(-1), axis=0)], # <--- inputs 
                     [np.expand_dims(FUTURE_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_CAMS*NUM_PAST_FRAMES), axis=0)])
        
        ms2 = time.time()*1000.0
        print("===================================================")
        print("iter: " + str(itrs) + " / timestep: "+str(t))
        print("Model update time: "+str(round((ms2-ms1)/1000, 3)) + ' secs')
        print("Current Model Loss: "+str(model_loss))
        MODEL_LOSS_BUFFER = np.roll(MODEL_LOSS_BUFFER, 1); MODEL_LOSS_BUFFER[0] = model_loss
        print(MODEL_LOSS_BUFFER); print("\n")
        
        ##### STEP 3: CHECKPOINTS #####
        
        if (itrs % IMAGE_COMPARE_CHECKPOINT == 0 ): # View how the model is doing
            print("Image compare checkpoint...")
            timestamp = str(time.time()).replace(".","")
            
            # Get the generated future frames: 
            ms1 = time.time()*1000.0
            GEN_FUTURE_FRAMES = UDM_model.predict(
                                [np.expand_dims(PREV_FRAMES_BUFFER.reshape(CAM_W, CAM_H, CAM_C*NUM_CAMS*NUM_PAST_FRAMES), axis=0), 
                                 np.expand_dims(PREV_ACTION_BUFFER.reshape(-1), axis=0), 
                                 np.expand_dims(FUTURE_ACTION_BUFFER.reshape(-1), axis=0)])
            GEN_FUTURE_FRAMES = GEN_FUTURE_FRAMES[0]
            ms2 = time.time()*1000.0
            print("Model prediction time: "+str(round((ms2-ms1)/1000, 3)) +' secs'+' for '+str(NUM_FUTURE_FRAMES)+' future frames.')
            
            imgs_gen = Video_Utils.ViewFutureFrames(GEN_FUTURE_FRAMES.reshape(NUM_FUTURE_FRAMES, CAM_W, CAM_H, CAM_C*NUM_CAMS))
            imgs_split = np.zeros((30, NUM_FUTURE_FRAMES*CAM_W, 3)) # goes in between to separate real/generated images
            imgs_gt = Video_Utils.ViewFutureFrames(FUTURE_FRAMES_BUFFER)
            #print(imgs_gt.shape); print(imgs_split.shape); print(imgs_gen.shape);
            imgs_compare = np.concatenate((imgs_gt, imgs_split, imgs_gen),axis=0)
            #print(imgs_compare.shape)
            #cv2.imshow('image',imgs_compare) 
            #cv2.waitKey(0)
            img_filename = 'sample_'+timestamp+'.png' 
            cv2.imwrite(img_filename,(imgs_compare*255 + 127.5))
            print("Wrote new image sample checkpoint at: "+img_filename)
            
        if (itrs % SAVE_CHECKPOINT_ITRS == 0): # save progress
            print("Model save checkpoint, itr: "+str(itrs))
            ms1 = time.time()*1000.0
            timestamp = str(time.time()).replace(".","")
            mean_recent_loss = round(np.mean(MODEL_LOSS_BUFFER), 4)
            model_str_name = 'UDM_weights_'+str(mean_recent_loss)+'.h5' # pro-tip: manually re-name after each run... 
            UDM_model.save(model_str_name)
            ms2 = time.time()*1000.0
            print("Model save time: "+str(round((ms2-ms1)/1000, 3)) +' secs')

        itrs = itrs + 1 # don't forget to increment total training itrs counter    


# In[ ]:




# In[ ]:




# In[ ]:



