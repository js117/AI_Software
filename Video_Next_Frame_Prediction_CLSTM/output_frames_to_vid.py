import numpy as np
import cv2
import os.path
import sys
from sys import platform

# This is a hack script to turn a bunch of jpg images into a video, because similar functionality didn't work on the Ubuntu platform with cv2... 

VIDEO_FRAMES_FOLDER_NAME = 'output_frames'
WIDTH = 112
HEIGHT = 112

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

# Get ready to create output videos: 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (2*WIDTH,HEIGHT))

print("Creating comparison video: ")
for file in files: 
	vis = cv2.imread(file)
	cv2.imshow('video', vis)
	cv2.waitKey(100) # play at ~10 fps
	# Note - in this case we already did the operation when writing the jpg images previously. 
	#vis = (vis * 255.0).astype('u1')) # don't forget actual videos can't be b/w 0 and 1
	out.write(vis)
	
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()