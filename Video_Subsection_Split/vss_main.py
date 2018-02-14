# STACK: - Anaconda3 (Python 3.5.2, Anaconda 4.2.0 (64-bit): C:\Program Files\Anaconda3
#		 - TensorFlow backend (see: C:\Users\JDS\.keras\keras.json)
#		 - pip install --upgrade --ignore-installed tensorflow

import time
import numpy as np
import cv2
import os.path
import sys
from sys import platform
from random import randint
from math import sqrt
import matplotlib.pyplot as plt # watch out on linux... 

# Project constants: file system, video settings
VIDEO_FRAMES_FOLDER_NAME = 'sequence_video_frames'
VIDEO_NAME = 'sample_sequence_downsampled.mp4'
WIDTH = 112 
HEIGHT = 112
NUM_CHANNELS = 3
global TIMESTEPS
TIMESTEPS = 30
minus1 = (np.array([-1]).astype('float32'))[0] # JUST

# View an image, resized: 
#cv2.namedWindow("img1", cv2.WINDOW_AUTOSIZE); cv2.imshow("img1", cv2.resize(cv2.imread(img_path1), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img2", cv2.WINDOW_AUTOSIZE); cv2.imshow("img2", cv2.resize(cv2.imread(img_path2), (224, 224))); cv2.waitKey(0)
#cv2.namedWindow("img3", cv2.WINDOW_AUTOSIZE); cv2.imshow("img3", cv2.resize(cv2.imread(img_path3), (224, 224))); cv2.waitKey(0)

### TEST PLAYING VIDEO, create frame database if not exist already: 

def PrepareVideoFilesForTraining(vid_name, vid_folder):
# e.g. vid_name = 'sample_sequence_downsampled.mp4'
#	   vid_folder = 'sequence_video_frames'
	if not os.path.exists(vid_folder):
		os.makedirs(vid_folder)

		cap = cv2.VideoCapture(vid_name)
		numFrames = 0
		while True:
			if cap.grab():
				flag, frame = cap.retrieve()
				numFrames = numFrames + 1
				#name = "%d.jpg"%numFrames
				name = "%d.jpg"%(time.time()*1000)
				cv2.imwrite(VIDEO_FRAMES_FOLDER_NAME+"/"+name, frame)     # save frame as JPEG file
				if not flag:
					continue
				else:
					cv2.imshow('video', frame)
			if cv2.waitKey(10) == 27:
				break

		cap.release()
		cv2.destroyAllWindows()

		print("Num frames: ", numFrames) # expected: 15 fps * 40 secs = 600 frames

	# Create training and testing data: approx. 70 - 30 
	curr_dir = os.getcwd()
	search_dir = ""
	if platform == "win32":
		search_dir = os.getcwd()+"\\"+VIDEO_FRAMES_FOLDER_NAME # WINDOWS
	else:
		search_dir = os.getcwd()+"/"+VIDEO_FRAMES_FOLDER_NAME # LINUX
	os.chdir(search_dir)
	files = filter(os.path.isfile, os.listdir(search_dir))
	files = [os.path.join(search_dir, f) for f in files] # add path to each file
	files.sort(key=lambda x: x) #os.path.getmtime(x))
	os.chdir(curr_dir)
	
	return files
	# Now our files are sorted chronologically (they were created in this order)
	# and we can put into training data, similar to below format:
	#(60000, 28, 28, 1) # train (num_examples, width, height, channels)
	#(10000, 28, 28, 1) # test (num_examples, width, height, channels)
	

def CreateVidFrameTensor(files, width, height, grayscale=1, div=1):
# grayscale is default color. Pass in 0 for gray. 
# div for scaling
# Create and return a tensor of video frame files that we can use in some neural network model
	X_ALL = []
	itr = 0
	print("Preparing video files for training...")
	for file in files:
		itr = itr + 1
		im = cv2.resize(cv2.imread(file, grayscale), (round(width/div), round(height/div)))
		#s = im.shape
		#print(s)
		#im = np.reshape(im, (s[2], s[0], s[1]))
		X_ALL.append(im) #default: WIDTH, HEIGHT, NUM_CHANNELS
		
	X_ALL = np.asarray(X_ALL); 
	X_ALL = X_ALL.astype('float32') / 255.
	print("Video files prepared. Shape: ")		
	print(X_ALL.shape)
	NumFrames = X_ALL.shape[0]
	
	return X_ALL, NumFrames
	
def LPF(x):
	n = len(x)
	y = [None] * n
	y[0] = x[0]
	y[1] = x[1]
	
	for i in range(2,n):
		y[i] = 0.25*(x[i] + 2*x[i-1] + x[i-2])
	
	return y
	
def SubSectionSplit(X, N):
	X_t_p_1 = X[1:N]
	X_t = X[0:N-1]
	X_diff_abs = np.abs(np.subtract(X_t_p_1, X_t))
	
	whole_mean = np.mean(X_diff_abs)
	whole_stds = np.std(X_diff_abs)
	means = [0] * (N-1)
	stds = [0] * (N-1)
	for i in range(0, N-1):
		means[i] = np.mean(X_diff_abs[i])
		#stds[i] = np.std(X_diff_abs[i])
	
	means = LPF( (means - whole_mean) / whole_stds)
	#stds = LPF(stds - whole_stds)
	
	
	return means
	
def CumulativeSumRewardSeries(v): # v is e.g. mean abs diffs of a video frame-by-frame; 
# Returns a potential reward sequence for each time-step, to train a cost function from raw sensory features
# Visually, this is a nice step graph with vals that start at 0 and eventually reach 1. 
	
	n = len(v)
	r = [0] * n
	vmin = np.min(v)
	
	for t in range(1,n):
		r[t] = r[t-1] + np.abs(v[t] - vmin)**2
	
	return r/np.max(r)

	
	
	
if __name__ == '__main__':
	print("Running program...")

	files = PrepareVideoFilesForTraining(VIDEO_NAME, VIDEO_FRAMES_FOLDER_NAME) # change constants at top of file

	VidFrames, NumFrames = CreateVidFrameTensor(files, WIDTH, HEIGHT, 0, 1)
	print("Number of frames: ",NumFrames)
	
	v = SubSectionSplit(VidFrames, NumFrames)
	
	r = CumulativeSumRewardSeries(v)
	
	# Plot for testing: 
	t = np.arange(0, NumFrames-1, 1)
	plt.figure(1)
	plt.subplot(211)
	plt.plot(t, v, 'b-')
	plt.subplot(212)
	plt.plot(t, r, 'r-')
	plt.show()
	
	print("Program finished.")
	
	