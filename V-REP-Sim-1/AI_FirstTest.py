# Using Anaconda 3.5
# 
# V-REP instructions: https://www.youtube.com/watch?v=SQont-mTnfM 
# http://developers-club.com/posts/268313/ 

import time
#ms1 = 0; ms2 = 0
#print("\n\n --- Loading ML libraries (about 10 secs)... --- \n\n\n")
#ms1 = time.time()*1000.0
#import tensorflow as tf
import numpy as np
import cv2
import os.path
import sys
import msvcrt # WINDOWS ONLY (what does Linux need?)
from sys import platform
from random import randint
from math import sqrt
#ms2 = time.time()*1000.0
#print("\nLoaded ML libraries. Time elasped: ", int(ms2-ms1), " ms\n")

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

def GetJointPosition(joint):
	e,p = vrep.simxGetJointPosition(clientID, joint, vrep.simx_opmode_streaming)
	return p
	
def StepJoint(joint, deg_step):
	new_pos = GetJointPosition(joint) + deg_step # TODO: limit checking
	return vrep.simxSetJointTargetPosition(clientID, joint, new_pos, vrep.simx_opmode_streaming)
	
def StepGripper(joint, direction, velocity, force=20): 
	velocity = velocity * direction
	err = vrep.simxSetJointTargetVelocity(clientID, joint, velocity, vrep.simx_opmode_streaming)
	err = vrep.simxSetJointForce(clientID, joint, force, vrep.simx_opmode_oneshot)
	gripper_openClose = direction
	#print("dir: "+str(direction)+" / err: "+str(err))
	return err
	
def SetJointsToAbsPos(arr):
	StepJoint(l_joint1, arr[0])
	StepJoint(l_joint2, arr[1])
	StepJoint(l_joint3, arr[2])
	StepJoint(l_joint4, arr[3])
	StepJoint(l_joint5, arr[4])
	StepJoint(l_joint6, arr[5])
	StepGripper(l_joint1, int(arr[6]), velocity=0.4) # note: this will just correspond to a direction, +1 or -1
	
def GetJointPositionsAll(printMe=0): 

	arr = [0] * 7
	e,l_joint1_pos = vrep.simxGetJointPosition(clientID, l_joint1, vrep.simx_opmode_streaming); arr[0] = l_joint1_pos
	e,l_joint2_pos = vrep.simxGetJointPosition(clientID, l_joint2, vrep.simx_opmode_buffer); arr[1] = l_joint2_pos
	e,l_joint3_pos = vrep.simxGetJointPosition(clientID, l_joint3, vrep.simx_opmode_buffer); arr[2] = l_joint3_pos
	e,l_joint4_pos = vrep.simxGetJointPosition(clientID, l_joint4, vrep.simx_opmode_buffer); arr[3] = l_joint4_pos
	e,l_joint5_pos = vrep.simxGetJointPosition(clientID, l_joint5, vrep.simx_opmode_buffer); arr[4] = l_joint5_pos
	e,l_joint6_pos = vrep.simxGetJointPosition(clientID, l_joint6, vrep.simx_opmode_buffer); arr[5] = l_joint6_pos
	e,l_joint7_pos = vrep.simxGetJointPosition(clientID, l_joint7, vrep.simx_opmode_buffer); arr[6] = l_joint7_pos
	#e,r_joint1_pos = vrep.simxGetJointPosition(clientID, r_joint1, vrep.simx_opmode_buffer); arr[7] = r_joint1_pos
	#e,r_joint2_pos = vrep.simxGetJointPosition(clientID, r_joint2, vrep.simx_opmode_buffer); arr[8] = r_joint2_pos
	#e,r_joint3_pos = vrep.simxGetJointPosition(clientID, r_joint3, vrep.simx_opmode_buffer); arr[9] = r_joint3_pos
	#e,r_joint4_pos = vrep.simxGetJointPosition(clientID, r_joint4, vrep.simx_opmode_buffer); arr[10] = r_joint4_pos
	#e,r_joint5_pos = vrep.simxGetJointPosition(clientID, r_joint5, vrep.simx_opmode_buffer); arr[11] = r_joint5_pos
	#e,r_joint6_pos = vrep.simxGetJointPosition(clientID, r_joint6, vrep.simx_opmode_buffer); arr[12] = r_joint6_pos
	#e,r_joint7_pos = vrep.simxGetJointPosition(clientID, r_joint7, vrep.simx_opmode_buffer); arr[13] = r_joint7_pos
	
	if (printMe):
		print("\n====================================\n")
		print("  joint 1: "+str(round(l_joint1_pos,3))+"\n")
		print("  joint 2: "+str(round(l_joint2_pos,3))+"\n")
		print("  joint 3: "+str(round(l_joint3_pos,3))+"\n")
		print("  joint 4: "+str(round(l_joint4_pos,3))+"\n")
		print("  joint 5: "+str(round(l_joint5_pos,3))+"\n")
		print("  joint 6: "+str(round(l_joint6_pos,3))+"\n")
		print("  joint 7: "+str(round(l_joint7_pos,3))+"\n")
		#print("Right joint 1: "+str(round(r_joint1_pos,3))+"\n")
		#print("Right joint 2: "+str(round(r_joint2_pos,3))+"\n")
		#print("Right joint 3: "+str(round(r_joint3_pos,3))+"\n")
		#print("Right joint 4: "+str(round(r_joint4_pos,3))+"\n")
		#print("Right joint 5: "+str(round(r_joint5_pos,3))+"\n")
		#print("Right joint 6: "+str(round(r_joint6_pos,3))+"\n")
		#print("Right joint 7: "+str(round(r_joint7_pos,3))+"\n")
		print("\n====================================\n")
		
	return arr
	
def GetAndDisplayAndSaveImage(curr_action, save_frame_action):
	if (user_isRecordingCurr == 1): # if we're recording, we'll want the (frame, action) pair
		# DISPLAY:
		# Note -- assumes all resolutions (W x H) are equal. 
		err, resolution, image1 = vrep.simxGetVisionSensorImage(clientID, cam1, 0, vrep.simx_opmode_streaming)
		err, resolution, image2 = vrep.simxGetVisionSensorImage(clientID, cam2, 0, vrep.simx_opmode_streaming)
		err, resolution, image3 = vrep.simxGetVisionSensorImage(clientID, cam3, 0, vrep.simx_opmode_streaming)
		err, resolution, image4 = vrep.simxGetVisionSensorImage(clientID, cam4, 0, vrep.simx_opmode_streaming)
		if err == vrep.simx_return_ok:
			# cam1
			img1 = np.array(image1,dtype=np.uint8)
			img1.resize([resolution[1],resolution[0],3])
			img1 = cv2.flip(img1,0)
			# cam2
			img2 = np.array(image2,dtype=np.uint8)
			img2.resize([resolution[1],resolution[0],3])
			img2 = cv2.flip(img2,0)
			# cam3
			img3 = np.array(image3,dtype=np.uint8)
			img3.resize([resolution[1],resolution[0],3])
			img3 = cv2.flip(img3,0)
			# cam4
			img4 = np.array(image4,dtype=np.uint8)
			img4.resize([resolution[1],resolution[0],3])
			img4 = cv2.flip(img4,0)
			
			# SAVE: 
			# we can save as e.g. 128x128x12 (4 cams, 3 channels each) or 256x256x3 using concat image below
			if (save_frame_action == 1):
				print(curr_action)
				saved_img_buffer = np.concatenate((img1,img2,img3,img4), axis=2)
				img_file_name = "%d_x"%(time.time()*1000)
				act_file_name = "%d_u"%(time.time()*1000)
				np.save(user_recording_folder+"/"+img_file_name, saved_img_buffer)
				np.save(user_recording_folder+"/"+act_file_name, curr_action)
			
			# Display: command window ;) 
			img = np.concatenate((np.concatenate((img1, img2),axis=1), np.concatenate((img3, img4),axis=1)), axis=0)
			
			#print(img[50:55,50:55,:])
			
			cv2.imshow('image',img)
			cv2.waitKey(1)
			
	
def TestController(radStep):
	currentMotor = np.random.randint(1,8) #1,...,7
	direction = 0
	if np.random.randint(1,3) == 1: # 1 or 2 w/ 50% probs
		direction = 1
	else:
		direction = -1
		
	if (currentMotor == 1):
		StepJoint(l_joint1, direction*radStep)
	elif (currentMotor == 2):
		StepJoint(l_joint2, direction*radStep)
	elif (currentMotor == 3):
		StepJoint(l_joint3, direction*radStep)
	elif (currentMotor == 4):
		StepJoint(l_joint4, direction*radStep)
	elif (currentMotor == 5):
		StepJoint(l_joint5, direction*radStep)
	elif (currentMotor == 6):
		StepJoint(l_joint6, direction*radStep)
	elif (currentMotor == 7):
		#StepJoint(l_joint7, direction*user_degStep)
		StepGripper(l_joint7, direction, velocity=0.4, force=20)
	
	curr_action = np.zeros((7,1))
	step_size = 0
	if (currentMotor == 7):
		step_size = 1 # gripper is just open/close
	else: 
		step_size = radStep
	curr_action[currentMotor-1] = direction * step_size
	
	return curr_action
	
global clientID
global l_joint1
global l_joint2
global l_joint3
global l_joint4
global l_joint5
global l_joint6
global l_joint7
#global r_joint1
#global r_joint2
#global r_joint3
#global r_joint4
#global r_joint5
#global r_joint6
#global r_joint7
#global user_currentArm
global user_currentMotor
global user_degStep
global user_radStep
global INIT_POS
global user_isRecordingCurr
global user_isRecordingPrev
global user_recordingFramesBuffer # place a limit on sequence length that we store before writing
global user_recording_folder
global user_actionJustTaken
global user_isControllerMode
global gripper_openClose


if __name__ == '__main__':
	print("Running program...")
	
	# V-REP Setup... 
	vrep.simxFinish(-1) # just in case, close all opened connections
	clientID = vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
	if clientID!=-1:
		print("Connected to V-REP remote API server.\n")
	else:
		print("Connection not successful")
		sys.exit("Error: could not connect")
	
	#vrep.simxSynchronous(clientID,True)
		
	print("Initializing joints...")
	err, l_joint1 = vrep.simxGetObjectHandle(clientID, 'UR5_joint1', vrep.simx_opmode_oneshot_wait); print(l_joint1)
	err, l_joint2 = vrep.simxGetObjectHandle(clientID, 'UR5_joint2', vrep.simx_opmode_oneshot_wait); print(l_joint2)
	err, l_joint3 = vrep.simxGetObjectHandle(clientID, 'UR5_joint3', vrep.simx_opmode_oneshot_wait); print(l_joint3)
	err, l_joint4 = vrep.simxGetObjectHandle(clientID, 'UR5_joint4', vrep.simx_opmode_oneshot_wait); print(l_joint4)
	err, l_joint5 = vrep.simxGetObjectHandle(clientID, 'UR5_joint5', vrep.simx_opmode_oneshot_wait); print(l_joint5)
	err, l_joint6 = vrep.simxGetObjectHandle(clientID, 'UR5_joint6', vrep.simx_opmode_oneshot_wait); print(l_joint6)
	err, l_joint7 = vrep.simxGetObjectHandle(clientID, 'RG2_openCloseJoint', vrep.simx_opmode_oneshot_wait); print(l_joint7)
	#err, r_joint1 = vrep.simxGetObjectHandle(clientID, 'r_joint1', vrep.simx_opmode_oneshot_wait); print(r_joint1)
	#err, r_joint2 = vrep.simxGetObjectHandle(clientID, 'r_joint2', vrep.simx_opmode_oneshot_wait); print(r_joint2)
	#err, r_joint3 = vrep.simxGetObjectHandle(clientID, 'r_joint3', vrep.simx_opmode_oneshot_wait); print(r_joint3)
	#err, r_joint4 = vrep.simxGetObjectHandle(clientID, 'r_joint4', vrep.simx_opmode_oneshot_wait); print(r_joint4)
	#err, r_joint5 = vrep.simxGetObjectHandle(clientID, 'r_joint5', vrep.simx_opmode_oneshot_wait); print(r_joint5)
	#err, r_joint6 = vrep.simxGetObjectHandle(clientID, 'r_joint6', vrep.simx_opmode_oneshot_wait); print(r_joint6)
	#err, r_joint7 = vrep.simxGetObjectHandle(clientID, 'r_joint7', vrep.simx_opmode_oneshot_wait); print(r_joint7)
	err, cam1 = vrep.simxGetObjectHandle(clientID, 'cam1', vrep.simx_opmode_oneshot_wait); print(cam1)
	err, cam2 = vrep.simxGetObjectHandle(clientID, 'cam2', vrep.simx_opmode_oneshot_wait); print(cam2)
	err, cam3 = vrep.simxGetObjectHandle(clientID, 'cam3', vrep.simx_opmode_oneshot_wait); print(cam3)
	err, cam4 = vrep.simxGetObjectHandle(clientID, 'servo_cam', vrep.simx_opmode_oneshot_wait); print(cam4)
	
	######## INIT POSITION ######## (confirm with the simulation where you want to start)
	INIT_POS = [-2.964, 0.083, -0.343, -0.627, 1.538, 0.259, 0]
	SetJointsToAbsPos(INIT_POS)
	gripper_openClose = 0 # will be same as direction variable for actions
	
	# Command time...
	#user_currentArm = 0 # 0 or 1, left or right arm respectively
	user_currentMotor = 1 # numbers 1,2,3,4,5,6,7 to change arms
	user_degStep = 2 # default: 2 degrees at a time per step. Can increase speed (and decrease for precision) w/ this param. 
	user_radStep = user_degStep/57.2957795 # radians per user step 
	direction = 0
	# Recording variables:
	user_isRecordingCurr = -1
	user_isRecordingPrev = -1
	user_actionJustTaken = 0
	user_isControllerMode = 0
	print("========== WELCOME TO APOLLO SIM V1. INSTRUCTIONS: ==========\n")
	print("\n=== z/x: change arms \n=== q/w: motor fwd/back \n=== 1/2/3/4/5/6/7: motor select \n=== a/s: speed up/down \n=== c: joint info \n=== r: start/stop recording \n=== t: toggle controller mode\n")
	print("=============================================================\n")
	
	vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
	
	while True: 

		if user_isControllerMode == 1:
			curr_action = TestController(user_radStep)
			GetAndDisplayAndSaveImage(curr_action, save_frame_action=0)
			user_actionJustTaken = 1
	
		#vrep.simxSynchronousTrigger(clientID)
		if user_actionJustTaken == 1:
			GetAndDisplayAndSaveImage(curr_action=0, save_frame_action=0)
			GetAndDisplayAndSaveImage(curr_action=0, save_frame_action=0)
			GetAndDisplayAndSaveImage(curr_action=0, save_frame_action=0)
			# ^ hacky, but empirically this seems to fix the visual update shown to the Python user. 
			# Was previously a few frames of delay after a command. 
			user_actionJustTaken = 0
		
		
		if msvcrt.kbhit():
			
			n = msvcrt.getch().decode("utf-8").lower()
			#print(n)
			# SETTINGS
			#if (n == 'z'):
			#	user_currentArm = 0
			#	print("Arm change: left\n")
			#if (n == 'x'):
			#	user_currentArm = 1
			#	print("Arm change: right\n")
			if (n == '1' or n == '2' or n == '3' or n == '4' or n == '5' or n == '6' or n == '7'):
				user_currentMotor = int(n)
				print("Motor change: "+str(user_currentMotor)+"\n")
			if (n == 'c'):
				arr = GetJointPositionsAll(printMe=1)
				print(arr)
			if (n == 'a'):
				user_degStep = user_degStep + 0.5
				if (user_degStep > 10): # some max degrees/step
					user_degStep = 10
				user_radStep = user_degStep/57.2957795
			if (n == 's'):
				user_degStep = user_degStep - 0.5
				if (user_degStep < 0.5):
					user_degStep = 0.5
				user_radStep = user_degStep/57.2957795
			# MOTION
			if (n == 'q' or n == 'w'): 
			
				if (n == 'q'):
					direction = 1
				elif (n == 'w'):
					direction = -1
				else:
					direction = 0 # shouldn't happen
				#if (user_currentArm == 0): # left arm
				if (user_currentMotor == 1):
					StepJoint(l_joint1, direction*user_radStep)
				elif (user_currentMotor == 2):
					StepJoint(l_joint2, direction*user_radStep)
				elif (user_currentMotor == 3):
					StepJoint(l_joint3, direction*user_radStep)
				elif (user_currentMotor == 4):
					StepJoint(l_joint4, direction*user_radStep)
				elif (user_currentMotor == 5):
					StepJoint(l_joint5, direction*user_radStep)
				elif (user_currentMotor == 6):
					StepJoint(l_joint6, direction*user_radStep)
				elif (user_currentMotor == 7):
					#StepJoint(l_joint7, direction*user_degStep)
					StepGripper(l_joint7, direction, velocity=0.4, force=20)

				else:
					print("Invalid user_currentArm and/or user_currentMotor\n")
					
				# CREATE ACTION VECTOR
				user_actionJustTaken = 1
				curr_action = np.zeros((7,1))
				step_size = 0
				if (user_currentMotor == 7):
					step_size = 1 # gripper is just open/close
				else: 
					step_size = user_radStep
				curr_action[user_currentMotor-1] = direction * step_size
				# DISPLAY, SAVE: 
				GetAndDisplayAndSaveImage(curr_action, save_frame_action=1)
								
					
			# /MOTION	

			# RECORDING
			if (n == 'r'):
				user_isRecordingCurr = user_isRecordingCurr * -1 # flip between -1 (off), 1 (on)
				if (user_isRecordingCurr == 1):
					print("\n === RECORDING SEQUENCE... ===\n")
					
					user_recording_folder =  "u_sequence_%d"%(time.time()*1000)
					os.makedirs(user_recording_folder)
					print("Folder name: " + user_recording_folder)
					
					#vrep.simxSynchronousTrigger(clientID)
					GetAndDisplayAndSaveImage(curr_action=0, save_frame_action=0)	
					
				if (user_isRecordingCurr == -1 and user_isRecordingPrev == 1): # they pressed 'r' while recording
					print("\n === STOPPED RECORDING. Sequence stats: ===\n")
					
				user_isRecordingPrev = user_isRecordingCurr
				
			# CONTROLLER MODE
			if (n == 't'):
				print("\n=== CONTROLLER MODE: ===\n")
				if (user_isControllerMode == 0):
					user_isControllerMode = 1
					user_isRecordingCurr = 1 # to get display
				else:
					user_isControllerMode = 0
					user_isRecordingCurr = 0
				
		

		
	
	