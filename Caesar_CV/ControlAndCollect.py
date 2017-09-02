# ControlAndCollect.py: 
#
# Allow the user to control the robot with serial commands, and record the vision sensor data + stream of commands

import cv2
import numpy as np
import time
import os
import serial
import sys

from camera import Camera
from draw import Gravity, put_text


from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes

############################## IMPORTANT GLOBAL VARIABLES ###############################
global RobotSerialController
global baudrate
global RobotCurrentJoint	# Integer from 1 to 8: 6 robot joints, + gripper, + neck
global RobotJointSpeeds
global JointMinSpeed
global JointMaxSpeed
global NumStepsPerPress
# Joint Name/Description by array index:		
# 0 - Base Rotation	
# 1 - Shoulder Pitch	
# 2 - Elbow Pitch	
# 3 - Elbow Roll	
# 4 - Wrist Pitch	
# 5 - Wrist Roll	
# 6 - Gripper Open/Close	
# 7 - Neck Pitch

### Some initializations: 
baudrate = 115200
RobotCurrentJoint = 0
RobotJointSpeeds = [100, 100, 100, 100, 100, 100, 100, 100]
JointMinSpeed = 10
JointMaxSpeed = 2500
NumStepsPerPress = 100

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

ports = serial_ports()
print(ports)
try:
	print("Connecting to port: "+ports[0])
	RobotSerialController = serial.Serial(ports[0], baudrate) 
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
	
sensor_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
WIDTH_color = sensor_kinect.color_frame_desc.Width
HEIGHT_color = sensor_kinect.color_frame_desc.Height
WIDTH_depth = sensor_kinect.depth_frame_desc.Width
HEIGHT_depth = sensor_kinect.depth_frame_desc.Height
WIDTH_ir = sensor_kinect.infrared_frame_desc.Width		# N.B. infrared is for seeing in the dark! 
HEIGHT_ir = sensor_kinect.infrared_frame_desc.Height

cv2.namedWindow("cam", cv2.WINDOW_NORMAL)

VIEW_MODE_RGB = 0
VIEW_MODE_DEPTH = 1
VIEW_MODE_IR = 2
VIEW_MODE = VIEW_MODE_RGB

RESIZED_WIDTH = 480.0
t1 = 0
t2 = 0

while True: 
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
		
		#print(c_frame.shape)
		#print(d_frame.shape)
		#print(ir_frame.shape)
		#print("------------------")
		
		
		
		

		
		

		# REAL-TIME QR CODE TRACKING: doesn't seem that it will work without sufficiently large pixel size 
		
		# However - we may be able to aid our computer vision algorithms by augmenting their input tensors with existing 
		# computer vision filters: edges, gradients, blobs, etc. This would let a CNN's lower-level filters move on to 
		# more interesting and task-specific stuff. Maybe it's like getting pre-trained low-level layers for free :) 
		
		# TODO: 
		# 1. Create "CV filtered inputs" of c_frame via edges, gradients, blobs, etc
		# 2. Create "command screen" composite image of c_frame with above, but also DEPTH and IR. Make 'composite view' into new mode.
		# 3. Python robot control, and therefore user input command recording. See "QA_Basic_Demo.py"
			
		
		#rgb_display_frame = cv2.drawKeypoints(c_frame_ds, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=None)
		
		
		#print(len(kp_fast))
		#print(kp_fast[0])
		t2 = time.time()*1000 - t1
		#print(t2)
		if (VIEW_MODE == VIEW_MODE_RGB):
			cv2.imshow("cam",c_frame_ds)
		elif (VIEW_MODE == VIEW_MODE_DEPTH):
			cv2.imshow("cam",d_frame)
		elif (VIEW_MODE == VIEW_MODE_IR):
			cv2.imshow("cam",ir_frame)
		else:
			print("Error: view mode undefined: " + str(VIEW_MODE))
			break
		
		
		wait_key = (cv2.waitKey(1) & 0xFF)
		
		if wait_key == ord('z'):	# QUIT PROGRAM
			break
		if wait_key == ord('v'):	# TOGGLE VIEW MODE
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

		if wait_key == ord('s'):
			timestamp = str(time.time()).replace(".","")
			img_name = timestamp+'.jpg'
			print("\nSaving image: "+img_name)
			cv2.imwrite(img_name, c_frame) # new_dataset_folder+"/"+
			continue

		if wait_key == ord('1'):
			RobotCurrentJoint = 0
			print("Manual control - current joint set to: BASE ROTATION \n")
			
		if wait_key == ord('2'):
			RobotCurrentJoint = 1
			print("Manual control - current joint set to: SHOULDER PITCH \n")
		
		if wait_key == ord('3'):
			RobotCurrentJoint = 2
			print("Manual control - current joint set to: ELBOW PITCH \n")
			
		if wait_key == ord('4'):
			RobotCurrentJoint = 3
			print("Manual control - current joint set to: ELBOW ROLL \n")
			
		if wait_key == ord('5'):
			RobotCurrentJoint = 4
			print("Manual control - current joint set to: WRIST PITCH \n")
			
		if wait_key == ord('6'):
			RobotCurrentJoint = 5
			print("Manual control - current joint set to: WRIST ROLL \n")
			
		if wait_key == ord('7'):
			RobotCurrentJoint = 6
			print("Manual control - current joint set to: GRIPPER \n")
			
		if wait_key == ord('8'):
			RobotCurrentJoint = 7
			print("Manual control - current joint set to: NECK \n")
			
		if wait_key == ord('q'):
			cmd_str = "MJ"+chr(65+RobotCurrentJoint)+str("0")+str(NumStepsPerPress)+"\n"; cmd_str = cmd_str.encode('utf-8')
			print(cmd_str)
			RobotSerialController.write(cmd_str)
			print("Manual control - move FORWARD \n")
			
		if wait_key == ord('w'):
			cmd_str = "MJ"+chr(65+RobotCurrentJoint)+str("1")+str(NumStepsPerPress)+"\n"; cmd_str = cmd_str.encode('utf-8')
			print(cmd_str)
			RobotSerialController.write(cmd_str)
			print("Manual control - move BACKWARD \n")

