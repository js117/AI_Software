import bpy
import math
import time

import sys
#import serial
import glob
#import cv2
#import tensorflow as tf



import os 

print("Executable: ")
print(os.path.dirname(sys.executable))
print(sys.executable)
print("Sys version: ")
print(sys.version)
print("os.__file__: ")
print(os.__file__)
# ^Use the above to figure out where Blender's python executable is. Then get pip and the libs needed.

# Setting up tips:
# Download 'get-pip.py' from online, run using Blender's Python executable.
# Then use pip.exe to install packages as needed. 
# e.g.: 
# [Open cmd.exe in admin mode] 
# cd C:\Program Files\Blender Foundation\Blender\2.79\python\bin
# python get_pip.py 
# cd C:\Program Files\Blender Foundation\Blender\2.79\python\Scripts
# pip.exe install opencv-python

# USING ANOTHER PYTHON INSTALL: 
# - Make sure it's the same as sys.version, e.g. Python 3.5
# - Try the following: 
# 	import sys
#	sys.path.append("C:/Python32/Lib/site-packages")
#	import numpy
#	print(dir(numpy))

# Viewing console from Blender: Window --> Toggle System Console

#port=''.join(glob.glob("/dev/ttyUSB*"))
#ser = serial.Serial(port,115200)
#print("connected to: " + ser.portstr)

#if os.name == 'nt': # Microsoft OS 
#    import msvcrt
    

# LIBRARY CONSIDERATIONS: 
#  msacm32.dll  avifil32.dll  avicap32.dll  msvfw32.dll
#

try:
    from msvcrt import getch  # try to import Windows version
except ImportError:
    def getch():   # define non-Windows version
        import termios, tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
 
def keypress():
    global char
    char = getch()



print("\nStarting script\n")

ob = bpy.data.objects['Armature']
bpy.context.scene.objects.active = ob
bpy.ops.object.mode_set(mode='POSE')

########### BONE OBJECTS ---> PRORGAMMABLE JOINTS ###########
# (for convenience we put the local rotation angle of the joint in the model 
#  in the variable name, i.e. Jn_{x,y,z}) 
base = ob.pose.bones['Bone']        # stationary base
J1_y = ob.pose.bones['Bone.001']      # base rotation / shoulder yaw
J2_y = ob.pose.bones['Bone.002']      # shoulder pitch
J3_y = ob.pose.bones['Bone.004']      # elbow pitch
J4_y = ob.pose.bones['Bone.007']      # elbow roll
J5_y = ob.pose.bones['Bone.008']      # wrist pitch
J6_y = ob.pose.bones['Bone.009']      # wrist roll

CurrentJoint = J1_y
CurrentJointLocalAxis = 'Y'
CurrentDegPerStep = 1
DegPerStepChangeFactor = 1.1

def get_local_orientation(pose_bone):
    local_orientation = pose_bone.matrix_channel.to_euler()
    if pose_bone.parent is None:
        return local_orientation
    else:
        x=local_orientation.x-pose_bone.parent.matrix_channel.to_euler().x
        y=local_orientation.y-pose_bone.parent.matrix_channel.to_euler().y
        z=local_orientation.z-pose_bone.parent.matrix_channel.to_euler().z
        return(x,y,z)
    
    
def step_angle(Joint, axis_str, degrees, dir): # dir is +/- 1
    # ex: 
    # step_angle(J1_y, 'Y', 5.2, -1)
    
    [x,y,z] = get_local_orientation(Joint)
    [x_new, y_new, z_new] = [i + dir*degrees for i in [x,y,z]]
    
    Joint.rotation_mode = 'XYZ'

    if axis_str == 'X':
        Joint.rotation_euler.rotate_axis(axis_str, math.radians(x_new))
    if axis_str == 'Y':
        Joint.rotation_euler.rotate_axis(axis_str, math.radians(y_new))
    if axis_str == 'Z':
        Joint.rotation_euler.rotate_axis(axis_str, math.radians(z_new))
                
    # Necessary?
    Joint.keyframe_insert(data_path="rotation_euler" ,frame=1)


def sendAngles_toArduino():
    #
    # Basically calculate some angles to send over serial to Arduino
    #
    #bone1=ob.pose.bones['Link1IK']
    #bone2=ob.pose.bones['Link2IK']
        
    #angle1=str(round(math.degrees(get_local_orientation(bone1)[2])+offset1))#[0]=x,[1]=y,[2]=z
    #angle2=str(round(math.degrees(get_local_orientation(bone2)[2])+offset2))
    angle1 = 0
    angle2 = 0    
    
    print( "%s  %s  \n" %( angle1, angle2 ) )

    #ser.write((angle1+','+angle2).encode('UTF-8'))


def frameChange(passedScene):
    print("frameChange\n")
    #sendAngles_toArduino()
    
bpy.app.handlers.frame_change_pre.append(frameChange)

############## JS testing below here ##############


while True:
    
    n = getch()
    if os.name == 'nt':
        n = n.decode('ASCII')
        
    print(n)
    
    if (n == '0'):
        print("Exiting user input mode.\n")
        break
        
    ################ JOINT SELECTION ###############
    if (n == '1'):
        print("--- Switching to joint 1: Base Rotation ---")
        CurrentJoint = J1_y
        CurrentJointLocalAxis = 'Y' 
    if (n == '2'):
        print("--- Switching to joint 2: Shoulder Pitch ---")
        CurrentJoint = J2_y
        CurrentJointLocalAxis = 'Y' 
    if (n == '3'):
        print("--- Switching to joint 3: Elbow Pitch ---")
        CurrentJoint = J3_y
        CurrentJointLocalAxis = 'Y' 
    if (n == '4'):
        print("--- Switching to joint 4: Elbow Roll ---")
        CurrentJoint = J4_y
        CurrentJointLocalAxis = 'Y' 
    if (n == '5'):
        print("--- Switching to joint 5: Wrist Pitch ---")
        CurrentJoint = J5_y
        CurrentJointLocalAxis = 'Y' 
    if (n == '6'):
        print("--- Switching to joint 6: Wrist Roll ---")
        CurrentJoint = J6_y
        CurrentJointLocalAxis = 'Y' 
        
    ################ SPEED/STEP SIZE ###############
    if (n == 'a'):
        CurrentDegPerStep = CurrentDegPerStep * DegPerStepChangeFactor 
        print("Increasing angle step size to (degress): "+str(CurrentDegPerStep))
    if (n == 's'):
        CurrentDegPerStep = CurrentDegPerStep / DegPerStepChangeFactor 
        print("Decreasing angle step size to (degress): "+str(CurrentDegPerStep))    
    
    
    ################ FORWARD JOINT MOTION ###############
    if (n == 'q'):
        print("Stepping current joint forward.")
        step_angle(CurrentJoint, CurrentJointLocalAxis, CurrentDegPerStep, 1)
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    if (n == 'w'):
        print("Stepping current joint backward.")
        step_angle(CurrentJoint, CurrentJointLocalAxis, CurrentDegPerStep, -1)
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)