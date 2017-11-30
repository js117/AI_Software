# ControlAndCollect.py: 
#
# Allow the user to control the robot with serial commands, and record the vision sensor data + stream of commands

import cv2
import numpy as np
import time
import os
import serial
import sys
import threading
import msvcrt # WINDOWS ONLY (what does Linux need?)
import pickle

from camera import Camera
from draw import Gravity, put_text

import pygame
from pygame.locals import *

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes

import array as ARR
from collections import deque
import matplotlib
from matplotlib import pyplot as plt
from pylab import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

############################## IMPORTANT GLOBAL VARIABLES ###############################
global RobotSerialController
global baudrate
global RobotCurrentJoint	# Integer from 1 to 8: 6 robot joints, + gripper, + neck
global RobotJointSpeeds
global JointMinSpeed
global JointMaxSpeed
global NumStepsPerPress
global CustomCommandString
global RecordingBufferComposite
global RecordingBufferExplicit
global LastRecordedMotionBuffer # for easy replay & reverse
# Joint Name/Description by array index:		
# 0 - Base Rotation	
# 1 - Shoulder Pitch	
# 2 - Elbow Pitch	
# 3 - Elbow Roll	
# 4 - Wrist Pitch	
# 5 - Wrist Roll	
# 6 - Gripper Open/Close	
# 7 - Neck Pitch

global IMUSerialControllers
global NumIMUs
global baudrate
global BytesPerBuffer
global plot_time_axes
global IMU_BUFFER_SIZE
global IMU_buffer_roll
global IMU_buffer_pitch
global IMU_buffer_yaw
global IMU_buffer_ax
global IMU_buffer_ay
global IMU_buffer_az
global IMU_buffer_gx
global IMU_buffer_gy
global IMU_buffer_gz
global IMU_buffer_mx
global IMU_buffer_my
global IMU_buffer_mz

### Some initializations: 
baudrate = 115200
RobotCurrentJoint = 0
RobotJointSpeeds = [400, 400, 400, 400, 400, 400, 400, 400]
RecordingBufferComposite = [[], [], [], [], [], [], [], [], []] # RecordingBufferComposite[RobotCurrentJoint].append(dir * NumStepsPerPress) for manual user cmds
RecordingBufferExplicit = [[],[],[],[]] # this one should be easier to use; Buffer[0] is timestamp
# RecordingBufferExplicit[1].append(RobotCurrentJoint); RecordingBufferExplicit[2].append(dir); RecordingBufferExplicit[3].append(NumStepsPerPress);
JointMinSpeed = 10
JointMaxSpeed = 2500
NumStepsPerPress = 10
CustomCommandString = ""

# IMU initializations below
NumIMUs = 2
BytesPerBuffer = 13*4 +2		# 13 float values being sent via Arduino, 4 bytes each; plus newline and carriage return (see Arduino code)
IMUSerialControllers = []

IMU_BUFFER_SIZE = 300
plot_time_axes = [i for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_roll = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)] 
IMU_buffer_pitch = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_yaw = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_ax = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_ay = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_az = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gx = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gy = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gz = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_mx = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_my = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_mz = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]

#########################################################################################

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
	
######################################## INITIAL SERIAL CONNECTIONS #################################################		
		
ports = serial_ports()
print(ports)

'''
try:
	print("Connecting to port: "+ports[0])
	RobotSerialController = serial.Serial(ports[0], baudrate, timeout=3) 
	print("Successfully connected.")
	# Programming usage:
	# 
	# RobotSerialController.write("serial_cmd_string");
	for i in range(len(RobotJointSpeeds)):
		# How this works: e.g. to do "SSA400" to set speed of base rotation (joint 'A' on the controller) to 400; 
		# Use chr(65+i) to get ASCII string of appropriate letter (65 == 'A' so we start from there)
		cmd_str = "SS"+chr(65+i)+str(RobotJointSpeeds[i])+"\n"
		print(cmd_str.encode('utf-8'))
		RobotSerialController.write(cmd_str.encode('utf-8'))
		#response = RobotSerialController.read(100) # just for some debugging, since the controller talks back
		#print(response)
	
except:
	print("Error: could not connect over serial port. Please check that the cable is plugged in. Try unplugging other USB serial devices.")
	sys.exit()
'''
	
###################################################################
for i in range(len(ports)):
	try:
		print("testing port: " + str(ports[i]))
		test_port = serial.Serial(ports[i], baudrate, timeout=3)
		print("connected.")
		
		bytes1 = test_port.read(BytesPerBuffer)
		bytes2 = test_port.read(BytesPerBuffer)
		bytes3 = test_port.read(BytesPerBuffer)
		lenb1 = len(bytes1)
		lenb2 = len(bytes2)
		lenb3 = len(bytes3)
		
		print("------------------")
		print(bytes1); print("--- len b1: " + str(lenb1))
		print(bytes2); print("--- len b2: " + str(lenb2))
		print(bytes3); print("--- len b3: " + str(lenb3))
		
		if lenb3 == BytesPerBuffer:
			print(str(ports[i]) + " is an IMU sensor port.")
			IMUSerialControllers.append(test_port)
			print("")
			
		else:
			print("Testing if COM port is Robot Controller: ")
			for ii in range(len(RobotJointSpeeds)):
				# How this works: e.g. to do "SSA400" to set speed of base rotation (joint 'A' on the controller) to 400; 
				# Use chr(65+i) to get ASCII string of appropriate letter (65 == 'A' so we start from there)
				cmd_str = "SS"+chr(65+ii)+str(RobotJointSpeeds[ii])+"\n"
				print(cmd_str.encode('utf-8'))
				test_port.write(cmd_str.encode('utf-8'))
				response = test_port.readline() # just for some debugging, since the controller talks back
				print(response)
				print("------ i: "+str(ii))
				
			if response:
				RobotSerialController = test_port
				print(str(ports[i]) + " is a Robot Controller port.")
		print("")
	
	except:
		print("failed to connect to port.")
	
	
######################################## /END/ INITIAL SERIAL CONNECTIONS #################################################			
	
def tryIndex(inData, char):
	res = 0
	try:
		res = inData.index(char)
	except:
		pass
	return res
	
def trySubString(inData, p1, p2):
	res = ""
	try:
		res = inData[p1 : p2]
	except:
		pass
	return res
	
# PRECONDITION: movement command string must specify all 8, even if no motion, i.e. zero steps. Example:
# inData = "MJA00B00C00D00E17864F00G00H00"	
def ReverseCustomMoveCommand(inData, debug_mode=0):
	
	motion_mode = 0
	if "Z1" in inData: 
		motion_mode = 1
	
	J1start = tryIndex(inData, 'A') #inData.index('A');
	J2start = tryIndex(inData, 'B') #inData.index('B');
	J3start = tryIndex(inData, 'C') #inData.index('C');
	J4start = tryIndex(inData, 'D') #inData.index('D');
	J5start = tryIndex(inData, 'E') #inData.index('E');
	J6start = tryIndex(inData, 'F') #inData.index('F');
	J7start = tryIndex(inData, 'G') #inData.index('G');
	J8start = tryIndex(inData, 'H') #inData.index('H');
	
	if debug_mode == 1:
		print("Received starts: ")
		print(J1start); print(J2start); print(J3start); print(J4start); 
		print(J5start); print(J6start); print(J7start); print(J8start); 
	
	J1dir = int(trySubString(inData, J1start + 1, J1start + 2))
	J2dir = int(trySubString(inData, J2start + 1, J2start + 2))
	J3dir = int(trySubString(inData, J3start + 1, J3start + 2))
	J4dir = int(trySubString(inData, J4start + 1, J4start + 2))
	J5dir = int(trySubString(inData, J5start + 1, J5start + 2))
	J6dir = int(trySubString(inData, J6start + 1, J6start + 2))
	J7dir = int(trySubString(inData, J7start + 1, J7start + 2))
	J8dir = int(trySubString(inData, J8start + 1, J8start + 2))
	
	if debug_mode == 1:
		print("Received dirs: ")
		print(J1dir); print(J2dir); print(J3dir); print(J4dir); 
		print(J5dir); print(J6dir); print(J7dir); print(J8dir); 
	
	J1step = int(trySubString(inData, J1start + 2, J2start))
	J2step = int(trySubString(inData, J2start + 2, J3start))
	J3step = int(trySubString(inData, J3start + 2, J4start))
	J4step = int(trySubString(inData, J4start + 2, J5start))
	J5step = int(trySubString(inData, J5start + 2, J6start))
	J6step = int(trySubString(inData, J6start + 2, J7start))
	J7step = int(trySubString(inData, J7start + 2, J8start))
	J8step = int(trySubString(inData, J8start + 2, len(inData)))
	
	if debug_mode == 1:
		print("Received steps: ")
		print(J1step); print(J2step); print(J3step); print(J4step); 
		print(J5step); print(J6step); print(J7step); print(J8step); 
		
	if J1dir == 1:
		J1dir = 0
	elif J1dir == 0:
		J1dir = 1
	if J2dir == 1:
		J2dir = 0
	elif J2dir == 0:
		J2dir = 1
	if J3dir == 1:
		J3dir = 0
	elif J3dir == 0:
		J3dir = 1
	if J4dir == 1:
		J4dir = 0
	elif J4dir == 0:
		J4dir = 1
	if J5dir == 1:
		J5dir = 0
	elif J5dir == 0:
		J5dir = 1
	if J6dir == 1:
		J6dir = 0
	elif J6dir == 0:
		J6dir = 1
	if J7dir == 1:
		J7dir = 0
	elif J7dir == 0:
		J7dir = 1
	if J8dir == 1:
		J8dir = 0
	elif J8dir == 0:
		J8dir = 1
		
	reverse_cmd = "MJ"; 
	if motion_mode == 1:
		reverse_cmd = reverse_cmd + "Z1"
	reverse_cmd = reverse_cmd + "A" + str(J1dir) + str(J1step)
	reverse_cmd = reverse_cmd + "B" + str(J2dir) + str(J2step)
	reverse_cmd = reverse_cmd + "C" + str(J3dir) + str(J3step)
	reverse_cmd = reverse_cmd + "D" + str(J4dir) + str(J4step)
	reverse_cmd = reverse_cmd + "E" + str(J5dir) + str(J5step)
	reverse_cmd = reverse_cmd + "F" + str(J6dir) + str(J6step)
	reverse_cmd = reverse_cmd + "G" + str(J7dir) + str(J7step)
	reverse_cmd = reverse_cmd + "H" + str(J8dir) + str(J8step)
	
	if debug_mode == 1:
		print("Original command: ")
		print(inData)
		print("Reverse command: ")
		print(reverse_cmd)
	
	return reverse_cmd
	#inData1 = "MJA01000B12000C03000D14000E05000F16000G17000H18000"	
	#inData2 = "MJA00B00C00D00E17864F00G00H00"
	#ReverseCustomMoveCommand(inData2, debug_mode=1)
	#sys.exit()


def ReplaySavedManualMotion(MotionBuffer, seq_dir=1):
	# RecordingBufferExplicit = [[timestamp],[selectedJoint],[direction],[numSteps]]
	numMoves = len(MotionBuffer[0])
	
	state = '0'
	
	if seq_dir == 1:
		for t in range(0, numMoves):
			currentJoint = MotionBuffer[1][t]
			currentDir = MotionBuffer[2][t]
			currentSteps = MotionBuffer[3][t]
			cmd_str = "MJ"+chr(65+currentJoint)+str(currentDir)+str(currentSteps)+"\n"; cmd_str = cmd_str.encode('utf-8')
			RobotSerialController.write(cmd_str)
			state = '0' # Write command to Arduino, set state to 0 until we get response. 
			# After a write, we should be able to read the updated state from Arduino
			while state == '0': # Wait for Arduino to be ready
				RobotSerialController.flushInput() # Clear input buffer
				state = str(RobotSerialController.read())
			
	elif seq_dir == -1:
		for t in range(numMoves-1, -1, -1):
			currentJoint = MotionBuffer[1][t]
			currentDir = MotionBuffer[2][t]
			newDir = -1
			if currentDir == 0:
				newDir = 1
			elif currentDir == 1:
				newDir = 0
			else:
				print("Error in recorded direction: " + str(currentDir))
				return
			currentSteps = MotionBuffer[3][t]
			cmd_str = "MJ"+chr(65+currentJoint)+str(newDir)+str(currentSteps)+"\n"; cmd_str = cmd_str.encode('utf-8')
			RobotSerialController.write(cmd_str)
			state = '0' # Write command to Arduino, set state to 0 until we get response. 
			# After a write, we should be able to read the updated state from Arduino
			while state == '0': # Wait for Arduino to be ready
				RobotSerialController.flushInput() # Clear input buffer
				state = str(RobotSerialController.read())
			
	else:
		print("Error: bad seq_dir: "+str(seq_dir))

		
		

	
	
sensor_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
WIDTH_color = sensor_kinect.color_frame_desc.Width
HEIGHT_color = sensor_kinect.color_frame_desc.Height
WIDTH_depth = sensor_kinect.depth_frame_desc.Width
HEIGHT_depth = sensor_kinect.depth_frame_desc.Height
WIDTH_ir = sensor_kinect.infrared_frame_desc.Width		# N.B. infrared is for seeing in the dark! 
HEIGHT_ir = sensor_kinect.infrared_frame_desc.Height

cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)

global is_exit
global is_recording_motions
global is_recording_composite # records motions and visual data
global pygame_screen
global VIEW_MODE
global c_frame
global d_frame
global ir_frame
global wrist_frame

VIEW_MODE_RGB = 0
VIEW_MODE_DEPTH = 1
VIEW_MODE_IR = 2
VIEW_MODE = VIEW_MODE_RGB

is_recording_motions = 0
is_recording_composite = 0

RESIZED_WIDTH = 480.0
t1 = 0
t2 = 0

def KeyPressThread(): 
	global VIEW_MODE
	global c_framew
	global d_frame
	global ir_frame
	global RobotSerialController
	global baudrate
	global RobotCurrentJoint	# Integer from 1 to 8: 6 robot joints, + gripper, + neck
	global RobotJointSpeeds
	global JointMinSpeed
	global JointMaxSpeed
	global NumStepsPerPress
	global CustomCommandString
	global pygame_screen
	global is_exit
	global is_recording_motions
	global is_recording_composite
	global LastRecordedMotionBuffer
	last_reverse_dir = -1
	
	pygame.init() 
	pygame_screen = pygame.display.set_mode((600, 450)) # this square corresponds to size of img we capture
	instructions_img = pygame.image.load('instructions_j1.jpg')
	
	state = '0' # for interacting with Arduino for serial commands
	
	while True:
		# Key press functionality
		if is_exit == 1:
			sys.exit()
			
		pygame_screen.blit(instructions_img,(0,0))
		pygame.display.flip()
		
		pygame.event.pump()
		keys = pygame.key.get_pressed()  #checking pressed keys
		
		# Doing it this way below gets "FAST KEYS", i.e. lots of responses if touched/held
		
		# MOVE FORWARDS
		if keys[pygame.K_q]: 
			cmd_str = "MJ"+chr(65+RobotCurrentJoint)+str("0")+str(NumStepsPerPress)+"\n"; cmd_str = cmd_str.encode('utf-8')
			#print(cmd_str)
			RobotSerialController.write(cmd_str)
			print("Manual control - move FORWARDS")
			
			state = '0' # Write command to Arduino, set state to 0 until we get response. 
			# After a write, we should be able to read the updated state from Arduino
			while state == '0': # Wait for Arduino to be ready
				RobotSerialController.flushInput() # Clear input buffer
				state = str(RobotSerialController.read())
			
			if is_recording_motions == 1:
				timestamp = time.time() * 1000
				RecordingBufferExplicit[0].append(timestamp)
				RecordingBufferExplicit[1].append(RobotCurrentJoint)
				RecordingBufferExplicit[2].append(0) # direction 
				RecordingBufferExplicit[3].append(NumStepsPerPress)
				
		# MOVE BACKWARDS
		if keys[pygame.K_w]: #k == ord('w'):
			cmd_str = "MJ"+chr(65+RobotCurrentJoint)+str("1")+str(NumStepsPerPress)+"\n"; cmd_str = cmd_str.encode('utf-8')
			#print(cmd_str)
			RobotSerialController.write(cmd_str)
			print("Manual control - move BACKWARDS")
			
			state = '0' # Write command to Arduino, set state to 0 until we get response. 
			# After a write, we should be able to read the updated state from Arduino
			while state == '0': # Wait for Arduino to be ready
				RobotSerialController.flushInput() # Clear input buffer
				state = str(RobotSerialController.read())
			
			if is_recording_motions == 1:
				timestamp = time.time() * 1000
				RecordingBufferExplicit[0].append(timestamp)
				RecordingBufferExplicit[1].append(RobotCurrentJoint)
				RecordingBufferExplicit[2].append(1) # direction 
				RecordingBufferExplicit[3].append(NumStepsPerPress)
				
		
		
		for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
				
					# Below: SLOW KEYS (i.e. capture 1 keypress
				
					if event.key == K_z: # QUIT PROGRAM
						is_exit = 1
						break
						
					if event.key == K_v: # TOGGLE VIEW MODE
						if (VIEW_MODE == VIEW_MODE_RGB):
							VIEW_MODE = VIEW_MODE_DEPTH
							continue
						elif (VIEW_MODE == VIEW_MODE_DEPTH):
							VIEW_MODE = VIEW_MODE_IR
							continue
						elif (VIEW_MODE == VIEW_MODE_IR):
							VIEW_MODE = VIEW_MODE_RGB
							continue
						else:
							print("Error: view mode undefined: " + str(VIEW_MODE))
							break
							
					if event.key == K_c: # CAPTURE IMAGE
						timestamp = str(time.time()).replace(".","")
						img_name = timestamp+'.jpg'
						print("\nSaving image: "+img_name)
						cv2.imwrite(img_name, c_frame) # new_dataset_folder+"/"+
						continue
						
					if event.key == K_1: 
						RobotCurrentJoint = 0
						instructions_img = pygame.image.load('instructions_j1.jpg')
						print("Manual control - current joint set to: BASE ROTATION \n")
						
					if event.key == K_2: 
						RobotCurrentJoint = 1
						instructions_img = pygame.image.load('instructions_j2.jpg')
						print("Manual control - current joint set to: SHOULDER PITCH \n")
					
					if event.key == K_3: 
						RobotCurrentJoint = 2
						instructions_img = pygame.image.load('instructions_j3.jpg')
						print("Manual control - current joint set to: ELBOW PITCH \n")
						
					if event.key == K_4: 
						RobotCurrentJoint = 3
						instructions_img = pygame.image.load('instructions_j4.jpg')
						print("Manual control - current joint set to: ELBOW ROLL \n")
						
					if event.key == K_5: 
						RobotCurrentJoint = 4
						instructions_img = pygame.image.load('instructions_j5.jpg')
						print("Manual control - current joint set to: WRIST PITCH \n")
						
					if event.key == K_6: 
						RobotCurrentJoint = 5
						instructions_img = pygame.image.load('instructions_j6.jpg')
						print("Manual control - current joint set to: WRIST ROLL \n")
						
					if event.key == K_7: 
						RobotCurrentJoint = 6
						instructions_img = pygame.image.load('instructions_j7.jpg')
						print("Manual control - current joint set to: GRIPPER \n")
						
					if event.key == K_8: 
						RobotCurrentJoint = 7
						instructions_img = pygame.image.load('instructions_j8.jpg')
						print("Manual control - current joint set to: NECK \n")
						
					# INCREASE JOINT SPEED
					if event.key == K_a:
						newSpeed = RobotJointSpeeds[RobotCurrentJoint] + 10
						if newSpeed >= JointMaxSpeed:
							newSpeed = JointMaxSpeed
						RobotJointSpeeds[RobotCurrentJoint] = newSpeed
						cmd_str = "SS"+chr(65+RobotCurrentJoint)+str(newSpeed)+"\n"; cmd_str = cmd_str.encode('utf-8')
						print(cmd_str)
						RobotSerialController.write(cmd_str)
						print("Increase speed of joint: " + str(RobotCurrentJoint+1) + " to: " + str(newSpeed))
						
					# DECREASE JOINT SPEED
					if event.key == K_s:
						newSpeed = RobotJointSpeeds[RobotCurrentJoint] - 10
						if newSpeed <= JointMinSpeed:
							newSpeed = JointMinSpeed
						RobotJointSpeeds[RobotCurrentJoint] = newSpeed
						cmd_str = "SS"+chr(65+RobotCurrentJoint)+str(newSpeed)+"\n"; cmd_str = cmd_str.encode('utf-8')
						print(cmd_str)
						RobotSerialController.write(cmd_str)
						print("Increase speed of joint: " + str(RobotCurrentJoint+1) + " to: " + str(newSpeed) + " \n")	
									
					if event.key == K_d: 
						NumStepsPerPress = NumStepsPerPress + 5
						print("Increase NumStepsPerPress to: "+str(NumStepsPerPress))	
						
					if event.key == K_f: 
						NumStepsPerPress = NumStepsPerPress - 5
						if NumStepsPerPress <= 5:
							NumStepsPerPress = 5
						print("Decrease NumStepsPerPress to: "+str(NumStepsPerPress))	
						
					if event.key == K_e:	# ENTER CUSTOM COMMAND STRING
						cmd_str = input("Enter your custom command string and press enter: ")
						CustomCommandString = cmd_str
						cmd_str = cmd_str + "\n"
						cmd_str = cmd_str.encode('utf-8')
						print(cmd_str)
						RobotSerialController.write(cmd_str)
						print("^ Running this custom command string. Press 'R' to reverse.")
						
					if event.key == K_r:	# REVERSE THE LAST CUSTOM COMMAND STRING
						cmd_str = ReverseCustomMoveCommand(CustomCommandString, debug_mode=1) # argument will hold the last custom command, by above logic. 
						# For now we are assuming speeds have been kept constant between the initial cmd and calling its reverse (does this matter for position?)
						CustomCommandString = cmd_str # for easy testing, allow us to keep reversing back and forth between 2 workspace positions
						cmd_str = cmd_str + "\n"
						cmd_str = cmd_str.encode('utf-8')
						print(cmd_str)
						RobotSerialController.write(cmd_str)
						print("^ Running this custom command string. Press 'R' to reverse.")	
							
					if event.key == K_t: 
						if is_recording_motions == 0: 
							RecordingBufferExplicit = [[],[],[],[]] # reset/clear buffer
							is_recording_motions = 1
							print("\n --- Recording Motions, press T again to stop, Y to stop and save. --- \n")
							break # back to main loop, doesn't exist program
						elif is_recording_motions == 1: 
							is_recording_motions = 0
							print("\n --- Stop recording motions. Not saved. --- \n")
							
					if event.key == K_y: 
						motion_to_save = input("Enter the name of your motion to save: ")
						with open(motion_to_save+".motion", 'wb') as fp:
							pickle.dump(RecordingBufferExplicit, fp)
						print("Saved motion as: " + motion_to_save+".motion" + "\n")
						LastRecordedMotionBuffer = RecordingBufferExplicit # make a copy
						is_recording_motions = 0
						last_reverse_dir = -1
						
					if event.key == K_g:
						print("\n --- Reversing last motion sequence... ---\n")
						print(RecordingBufferExplicit)
						ReplaySavedManualMotion(RecordingBufferExplicit, seq_dir = last_reverse_dir)
						# So after the first time a motion is saved, it will be reversed, as set in the motion_save above
						last_reverse_dir = last_reverse_dir * -1
							
						
		
		'''if msvcrt.kbhit():
			n = msvcrt.getch().decode("utf-8").lower()
			print(n)
			
				
			############## CURRENT TODO: FIX CUSTOM COMMANDS BELOW ####################	
				
			if n == 'e':	# ENTER CUSTOM COMMAND STRING
				cmd_str = input("Enter your custom command string and press enter: ")
				CustomCommandString = cmd_str
				cmd_str = cmd_str + "\n"
				cmd_str = cmd_str.encode('utf-8')
				print(cmd_str)
				RobotSerialController.write(cmd_str)
				print("^ Running this custom command string. Press 'R' to reverse.")
				
			if n == 'r':	# REVERSE THE LAST CUSTOM COMMAND STRING
				cmd_str = ReverseCustomMoveCommand(CustomCommandString, debug_mode=1) # argument will hold the last custom command, by above logic. 
				# For now we are assuming speeds have been kept constant between the initial cmd and calling its reverse (does this matter for position?)
				CustomCommandString = cmd_str # for easy testing, allow us to keep reversing back and forth between 2 workspace positions
				cmd_str = cmd_str + "\n"
				cmd_str = cmd_str.encode('utf-8')
				print(cmd_str)
				RobotSerialController.write(cmd_str)
				print("^ Running this custom command string. Press 'R' to reverse.")
		'''		
	sys.exit() # if breaking from while loop
	

keypress_thread = threading.Thread(target=KeyPressThread)
try:
	keypress_thread.setDaemon(True)  # important for cleanup ? 
	keypress_thread.start() # join too? 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()

is_exit = 0

wrist_cap = cv2.VideoCapture(1)

while True: 
	# NECK SENSOR
	if sensor_kinect.has_new_color_frame():
		t1 = time.time() * 1000
		c_frame = sensor_kinect.get_last_color_frame()
		c_frame = c_frame.reshape(HEIGHT_color, WIDTH_color, -1) 
		c_frame = c_frame[:,:,0:3] # it's too big
		#print(c_frame.shape)
		r = RESIZED_WIDTH / c_frame.shape[0]
		dim = (int(RESIZED_WIDTH), int(c_frame.shape[1] * r))
		#print(dim)
		c_frame_ds = cv2.resize(c_frame, dim, interpolation = cv2.INTER_AREA)
		
		d_frame = sensor_kinect.get_last_depth_frame(); d_len = d_frame.shape[0]
		
		d_frame = d_frame.reshape(HEIGHT_depth, WIDTH_depth, -1) / 4000
		ir_frame = sensor_kinect.get_last_infrared_frame(); ir_len = ir_frame.shape[0]
		ir_frame = ir_frame.reshape(HEIGHT_ir, WIDTH_ir, -1)
		

		t2 = time.time()*1000 - t1
		#print(t2)
		if (VIEW_MODE == VIEW_MODE_RGB):
			cv2.imshow("cam",c_frame)
		elif (VIEW_MODE == VIEW_MODE_DEPTH):
			cv2.imshow("cam",d_frame)
		elif (VIEW_MODE == VIEW_MODE_IR):
			cv2.imshow("cam",ir_frame)
		else:
			print("Error: view mode undefined: " + str(VIEW_MODE))
			break
	
	### WRIST SENSOR
	ret, wrist_frame = wrist_cap.read()
	if ret==True:
		wrist_frame = cv2.flip(wrist_frame,0)
		cv2.imshow("wrist", wrist_frame)
		
		
	wait_key = (cv2.waitKey(1) & 0xFF)
		
	if is_exit == 1:
		break
			
sys.exit()
		

			
			

