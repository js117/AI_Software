# Setup_Motors.py - a command line program to configure and test the various joints of the Apollo Robot Platform. 
# 
# Instructions:
# Run program, which will allow you to test each joint (random order in general) one-by-one. As you identify which
# joint is being tested, enter its unique 2-character name into the command prompt and hit enter. This will save the
# COM-port <---> Joint configuration settings into a file that should be consistent for the same wiring connector
# and USB plug settings. If either changes or problems arise on a particular run, just quickly re-run this script
# to re-assign the ports. 
import numpy as np
import cv2
import os.path
import sys
import msvcrt # WINDOWS ONLY 
from sys import platform
from random import randint
from math import sqrt
import serial
import glob
import pygame
from pygame.locals import *
import threading
import time

global ports
global baudrate
global CONFIG_FILENAME
global q,w,e,r,t,y

baudrate = 19200
CONFIG_FILENAME = 'APOLLO_MOTOR_CONFIG.txt' 
ports = []



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

print("\n==========================================================\n")
print("Configure available ports by testing motors one at a time.")
print("Assign each port to a motor, press the following if the current port corresponds with the motor: \n")
print("(Press 'q' and 'w' to test a motor, and 'e' to enter its unique 2-character code to the config.)\n")
print("\n APOLLO ACTUATOR NAMING GUIDE: \n")
print("L1 - left base rotation\n")
print("L2 - left base rotation\n")
print("L3 - left base rotation\n")
print("L4 - left base rotation\n")
print("L5 - left base rotation\n")
print("L6 - left base rotation\n")
print("R1 - left base rotation\n")
print("R2 - left base rotation\n")
print("R3 - left base rotation\n")
print("R4 - left base rotation\n")
print("R5 - left base rotation\n")
print("R6 - left base rotation\n")
print("N1 - left base rotation\n")
print("N2 - left base rotation\n")
print("\n==========================================================\n")
ports = [str(p) for p in serial_ports()]
print("--- Available ports: ---")
print(ports)
print("------------------------\n")

################################################ RAW TESTING (TODO: ORGANIZE ) #########################################################
global MOTOR_shoulder_yaw 
global MOTOR_shoulder_pitch 
global MOTOR_elbow_pitch 
global MOTOR_wrist_pitch
global MOTOR_wrist_roll
global MOTOR_gripper 
global MOTOR_left_shoulder_yaw
global q,w,e,r,t,y
q = "q"; q = q.encode('utf-8')
w = "w"; w = w.encode('utf-8')
t = "t"; t = t.encode('utf-8')
y = "y"; y = y.encode('utf-8')
e = "e"; e = e.encode('utf-8')
r = "r"; r = r.encode('utf-8')

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
			if (n == '1'):
				MOTOR_test.close() 
				MOTOR_shoulder_yaw = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as base rotation joint\n"); break
			if (n == '2'):
				MOTOR_test.close() 
				MOTOR_shoulder_pitch = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as shoulder pitch joint\n"); break
			if (n == '3'):
				MOTOR_test.close() 
				MOTOR_elbow_pitch = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as elbow joint\n"); break
			if (n == '4'):
				MOTOR_test.close() 
				MOTOR_wrist_pitch = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as wrist pitch joint\n"); break
			if (n == '5'):
				MOTOR_test.close() 
				MOTOR_wrist_roll = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as wrist roll joint\n"); break
			if (n == '6'):
				MOTOR_test.close() 
				MOTOR_gripper = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as gripper joint\n"); break
			if (n == '7'):
				MOTOR_test.close() 
				MOTOR_left_shoulder_yaw = serial.Serial(port, baudrate) 
				print("Port "+port+" selected as left base rotation joint\n"); break
print("\nFinished configuring ports.\n")

pygame.init() 	
screen = pygame.display.set_mode((200,200))
while(True):
		pygame.event.pump()
		keys = pygame.key.get_pressed()  #checking pressed keys

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
		if keys[pygame.K_t]: #k == ord('z'):
			MOTOR_wrist_roll.write(q); time.sleep(.001)
		if keys[pygame.K_y]: #k == ord('x'):
			MOTOR_wrist_roll.write(w); time.sleep(.001)
		if keys[pygame.K_g]: #k == ord('z'):
			MOTOR_gripper.write(q); time.sleep(.001)
		if keys[pygame.K_h]: #k == ord('x'):
			MOTOR_gripper.write(w); time.sleep(.001)
		if keys[pygame.K_b]: #k == ord('z'):
			MOTOR_left_shoulder_yaw.write(q); time.sleep(.001)
		if keys[pygame.K_n]: #k == ord('x'):
			MOTOR_left_shoulder_yaw.write(w); time.sleep(.001)
		
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
			
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
					
				# --
				if event.key == K_u:
					MOTOR_wrist_roll.write(t)
				if event.key == K_i:  
					MOTOR_wrist_roll.write(y)
					
				if event.key == K_j:
					MOTOR_gripper.write(t)
				if event.key == K_k:  
					MOTOR_gripper.write(y)
					
				if event.key == K_m:
					MOTOR_left_shoulder_yaw.write(t)
				if event.key == K_l:  
					MOTOR_left_shoulder_yaw.write(y)
					
				# Control step size:                 
				#if event.key == K_5:
				#	MOTOR_shoulder_yaw.write(e)
				#if event.key == K_6:  
				#	MOTOR_shoulder_yaw.write(r)
					
				#if event.key == K_t:
				#	MOTOR_shoulder_pitch.write(e)
				#if event.key == K_y:  
				#	MOTOR_shoulder_pitch.write(r)
					
				#if event.key == K_g:
				#	MOTOR_elbow_pitch.write(e)
				#if event.key == K_h:  
				#	MOTOR_elbow_pitch.write(r)
					
				#if event.key == K_b:
				#	MOTOR_wrist_pitch.write(e)
				#if event.key == K_n:  
				#	MOTOR_wrist_pitch.write(r)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
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
			if (n == 'e'):
				
				joint_str = str(input("Enter mode: input the 2-character joint name."))
				print("You entered: /// "+joint_str+" /// ")
				
				is_correct = str(input("Enter 'y' if this is correct, or 'n' if not."))
				if (is_correct != 'y'):
					break 
				
				with open(CONFIG_FILENAME, "a") as myfile:
					myfile.write(joint_str+'/'+port+'\n')
				
				MOTOR_test.close() 
				print("Port "+port+" selected as: "+joint_str) 
				break
				
myfile.close()				
print("\nFinished configuring ports.\n")
test_for_fun = str(input("Enter 'y' if you would like to test right now through command line, or 'q' to quit."))