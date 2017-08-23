#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Source:
# https://github.com/Kinect/PyKinect2/tree/master/examples 

import cv2
import numpy as np
import time
import os

from camera import Camera
from draw import Gravity, put_text


from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes

##################################### SETUP ######################################


sift_features = cv2.xfeatures2d.SIFT_create()
orb_features = cv2.ORB_create(nfeatures=250) # SPEED PERFORMANCE: 2.5x faster than SIFT
# http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#orb-orb
# SIFT rotation invariance explained, once and for all: 
'''
Rotation dependence The feature vector uses gradient orientations. Clearly, if you rotate the image, 
everything changes. All gradient orientations also change. To achieve rotation independence, the keypoint's
rotation is subtracted from each orientation. Thus each gradient orientation is relative to the keypoint's
 orientation. [that was bothering me]
 ''' 
cv2.ocl.setUseOpenCL(False) # Apparently this is needed to fix ORB and FLANN problems in OpenCV... 
QR_code_folder = "QR_codes"
QR_code_names = []
QR_code_imgs = []
QR_code_kps = []
QR_code_des = []
NUM_QR_CODES = 0

bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


print("Initializing QR codes...\n")
QR_folder_fullpath = os.getcwd()+"/"+QR_code_folder
for filename in os.listdir(QR_folder_fullpath):
	filepath = QR_folder_fullpath+"\\"+filename
	print(filepath)
	img = cv2.imread(filepath, 0)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kp, des = orb_features.detectAndCompute(img,None)
	QR_code_names.append(filename)
	QR_code_imgs.append(img)
	QR_code_kps.append(kp)
	QR_code_des.append(des)
	NUM_QR_CODES = NUM_QR_CODES + 1

	
		


	
	
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

RESIZED_WIDTH = 550.0
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
		
		
		
		

		
		#img3 = cv2.drawMatchesKnn(QR_code_imgs[i],kp1,c_frame_ds,kp2,matches,None,**draw_params)
		img3 = cv2.drawMatchesKnn(QR_code_imgs[i],kp1,c_frame_ds,kp2,good,flags=2,outImg=None)

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
		print(t2)
		if (VIEW_MODE == VIEW_MODE_RGB):
			cv2.imshow("cam",img3)
		elif (VIEW_MODE == VIEW_MODE_DEPTH):
			cv2.imshow("cam",d_frame)
		elif (VIEW_MODE == VIEW_MODE_IR):
			cv2.imshow("cam",ir_frame)
		else:
			print("Error: view mode undefined: " + str(VIEW_MODE))
			break
		
		
		wait_key = (cv2.waitKey(1) & 0xFF)
		
		if wait_key == ord('q'):	# QUIT PROGRAM
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

		



'''
def main():
    def callback(frame, depth, fps):
        # Normalize the depth for representation
        min, max = depth.min(), depth.max()
        depth = np.uint8(255 * (depth - min) / (max - min))

        # Unable to retrieve correct frame, it's still depth here
        put_text(frame, "{1}x{0}".format(*frame.shape), Gravity.TOP_LEFT)
        put_text(depth, "{1}x{0}".format(*depth.shape), Gravity.TOP_LEFT)

        put_text(depth, "%.1f" % fps, Gravity.TOP_RIGHT)

        cv2.imshow('frame', frame)
        cv2.imshow('depth', depth)

    with Camera(cv2.CAP_OPENNI2) as cam:
        print("Camera: %dx%d, %d" % (
            cam.get(cv2.CAP_OPENNI_IMAGE_GENERATOR + cv2.CAP_PROP_FRAME_WIDTH),
            cam.get(cv2.CAP_OPENNI_IMAGE_GENERATOR + cv2.CAP_PROP_FRAME_HEIGHT),
            cam.get(cv2.CAP_OPENNI_IMAGE_GENERATOR + cv2.CAP_PROP_FPS)))
        cam.capture(callback, False)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


# Using Kinect and other OpenNI compatible depth sensors:
#   http://docs.opencv.org/master/d7/d6f/tutorial_kinect_openni.html
# OpenCV Python unable to access correct OpenNI device channels:
#   https://github.com/opencv/opencv/issues/4735
'''

''' 
# Using pre-made CV features to do template matching
kp2, des2 = orb_features.detectAndCompute(c_frame_ds,None) 
		
		i = 1
		#for i in range(0,1): 
		kp1 = QR_code_kps[i] 
		des1 = QR_code_des[i]
		
		matches = bf_matcher.knnMatch(des1,des2,k=2)

		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.81*n.distance:
				good.append([m])
'''