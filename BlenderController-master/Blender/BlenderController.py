import bpy
import math
import time

import sys
import serial
import glob
import msvcrt

# Setting up tips:
# Download 'get-pip.py' from online, run using Blender's Python executable.
# Then use pip.exe to install packages as needed. 

# Accessing Blender Python console: Shift + F4
# Viewing console from Blender: Window --> Toggle System Console

#port=''.join(glob.glob("/dev/ttyUSB*"))
#ser = serial.Serial(port,115200)
#print("connected to: " + ser.portstr)

# TODO: look at extending script with user input, e.g.
# https://stackoverflow.com/questions/19554023/how-to-capture-keyboard-input-in-blender-using-python 

print("\nStarting script\n")

ob = bpy.data.objects['Armature']
bpy.context.scene.objects.active = ob
bpy.ops.object.mode_set(mode='POSE')

offset1=30
offset2=140



def get_local_orientation(pose_bone):
    local_orientation = pose_bone.matrix_channel.to_euler()
    if pose_bone.parent is None:
        return local_orientation
    else:
        x=local_orientation.x-pose_bone.parent.matrix_channel.to_euler().x
        y=local_orientation.y-pose_bone.parent.matrix_channel.to_euler().y
        z=local_orientation.z-pose_bone.parent.matrix_channel.to_euler().z
        return(x,y,z)


def sendAngles():
	
	bone1=ob.pose.bones['Link1IK']
	bone2=ob.pose.bones['Link2IK']
	    
	angle1=str(round(math.degrees(get_local_orientation(bone1)[2])+offset1))#[0]=x,[1]=y,[2]=z
	angle2=str(round(math.degrees(get_local_orientation(bone2)[2])+offset2))
    
	print( "%s  %s  \n" %( angle1, angle2 ) )

	#ser.write((angle1+','+angle2).encode('UTF-8'))





def frameChange(passedScene):
	print("frameChange\n")
	sendAngles()
    
bpy.app.handlers.frame_change_pre.append(frameChange)

############## JS testing below here ##############
joint1=ob.pose.bones['Link1FK']
joint2=ob.pose.bones['Link2FK']
# Set rotation mode to Euler XYZ, easier to understand
# than default quaternions
joint1.rotation_mode = 'XYZ'
joint2.rotation_mode = 'XYZ'



while True:
    if msvcrt.kbhit():
        n = msvcrt.getch().decode("utf-8").lower()
        if (n == '0'):
            print("Exiting user input mode.\n")
            break
        if (n == '1'):
            print("Received input 1\n")
        if (n == '2'):
            print("Received input 2\n")
        if (n == '3'):
            print("Received input 3\n")
        if (n == '4'):
            print("Received input 4\n")