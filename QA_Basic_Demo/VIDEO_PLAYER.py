# STACK: - Anaconda3 (Python 3.5.2, Anaconda 4.2.0 (64-bit): C:\Program Files\Anaconda3
#		 - TensorFlow backend (see: C:\Users\JDS\.keras\keras.json)
#		 - pip install --upgrade --ignore-installed tensorflow

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os.path
import sys
import platform

global WIDTH 
global HEIGHT

def GetDataFromFolder(folder, do_normalize, w, h):
	
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
	imgs = np.zeros((n, w, h, 3), dtype=np.uint8)
	
	i = 0
	for f in files:
		img = cv2.resize(cv2.imread(f), (w, h))
		imgs[i,:,:,:] = img
		i = i + 1
		
	if (do_normalize == 1):	
		imgs = (imgs.astype(float) - 127.5) / 255 # normalize	
	return imgs
	
def PlayVideoInCV2():
	cap = cv2.VideoCapture('sample_sequence_downsampled.mp4')
	numFrames = 0
	while True:
		if cap.grab():
			flag, frame = cap.retrieve()
			numFrames = numFrames + 1
			name = "%d.jpg"%numFrames
			cv2.imwrite(VIDEO_FRAMES_FOLDER_NAME+"/"+name, frame)     # save frame as JPEG file
			if not flag:
				continue
			else:
				cv2.imshow('video', frame)
		if cv2.waitKey(10) == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
###############################################################################################################

if __name__ == "__main__":

	VIDEO_FRAMES_FOLDER_NAME = str(sys.argv[1]) # precondition: should already exist
	WIDTH = int(sys.argv[2])
	HEIGHT = int(sys.argv[3])
	do_normalize = 0
	
	X_ALL = GetDataFromFolder(VIDEO_FRAMES_FOLDER_NAME, do_normalize, WIDTH, HEIGHT) #n,W,H,3

	print("X_ALL shape: ",X_ALL.shape)

	num_frames = X_ALL.shape[0]
	prev_frame = np.array([X_ALL[0,:,:,:]]) # to make shapes work initially.. 
	RECREATED_FRAMES = np.zeros(X_ALL.shape)

	print("\n Preparing sample output video...\n")
	

	# Get ready to create output videos: 
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 24.0, (WIDTH,HEIGHT))
		
	print("Creating comparison video: ")
	for i in range(0, num_frames):
		img1 = X_ALL[i,:,:,:]
		vis = img1
		cv2.imshow('video', vis)
		cv2.waitKey(33) # play at ~30 fps
		if (do_normalize == 1):
			out.write(((vis * 255.0)+127.5).astype('u1')) # don't forget actual videos can't be b/w 0 and 1
		else:
			out.write((vis).astype('u1')) # should work for UN-NORMALIZED frames
		
	out.release()
	cv2.destroyAllWindows()

'''

'''