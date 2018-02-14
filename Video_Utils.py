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

########################## Video_Utils.py ########################
# Use this file to do the following:
# - Interact with webcams (USB, laptop, etc)
# - Save image files to a folder (timestamped)
# - Load image files from a folder (in chronological order based on timestamp)
# - Normalize/prepare image data to be consumable by Keras neural networks (i.e. CNNs)
# - Create videos from folders that contain images (assumed to be timestamped)

# """image""" is a (N)xWxHx3 numpy array from a loaded image, in int format; N is optional, can be 3 or 4 tensor
# """output""": (N)xWxHx3 float tensor (numpy array) that has been appoximate mean subtracted ( - 127.5, i.e. half range),
#				and divided by max image range (i.e. 255) so all values are between [0, 1]
def NormalizeImgTensor(image): 
	output = (image.astype(float) - 127.5) / 255 # normalize
	return output
	
# For a raw un-normalized 3-Tensor image (WxHx3), normalize and expand dims so that it is directly consumable by a Keras model. 
# i.e. we can call myCNNmodel.predict(NormalizeImgAndExpandDims(image))	
def NormalizeImgAndExpandDims(image):
	output = np.expand_dims(NormalizeImgTensor(image), axis=0)
	return output
	
# Undo normalization, and convert datatype back to int	
def UnnormalizeImgTensor(img_tensor):
	output = ((img_tensor * 255.0)+127.5).astype(int) # 'ul' previously
	return output
	
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
	imgs = np.zeros((n, h, w, 3), dtype=np.uint8)
	
	i = 0
	for f in files:
		img = cv2.imread(f) #cv2.resize(cv2.imread(f), (w, h))
		imgs[i,:,:,:] = img
		i = i + 1
		
	if (do_normalize == 1):	
		imgs = NormalizeImgTensor(imgs)
	return imgs
	
def WriteFrameToFolder(recording_folder, frame):
	timestamp = str(time.time()).replace(".","")
	img_name = timestamp+'.jpg'
	cv2.imwrite(recording_folder+"/"+img_name, frame)	
	return
	
# Convert an actual video (e.g. .mp4 file) to a folder with video frame images 	
# Returns: number of frames that were created
def FromVideoCreateImagesFolder(vid_file_path, vid_frames_folder_name, play_vid):
	cap = cv2.VideoCapture(vid_file_path)
	num_frames = 0
	while True:
		if cap.grab():
			flag, frame = cap.retrieve()
			timestamp = str(time.time()).replace(".","")
			img_name = timestamp+'.jpg'
			cv2.imwrite(vid_frames_folder_name+"/"+img_name, frame)     # save frame as JPEG file
			num_frames = num_frames + 1
			if not flag:
				continue
			if play_vid == 1:
				cv2.imshow('video', frame)
				if cv2.waitKey(10) == 27:
					break

	if play_vid == 1:
		cap.release()
		cv2.destroyAllWindows()
		
	return num_frames
	
# image_tensor is a (N, W, H, 3) image tensor	
# output_file_name e.g. 'output.avi' (can include folder path too)	
def CreateOutputVideo(img_tensor, output_file_name, output_frame_rate, w, h, do_un_normalize, play_vid):
	# Get ready to create output videos: 
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_file_name, fourcc, output_frame_rate, (w,h))
	num_frames = img_tensor.shape[0]
	
	print("Creating comparison video: ")
	for i in range(0, num_frames):
		img1 = img_tensor[i,:,:,:]
		vis = img1
		cv2.imshow('video', vis)
		cv2.waitKey(round(1000/output_frame_rate)) # play at approx. output_frame_rate
		if (do_un_normalize == 1):
			out.write(UnnormalizeImgTensor(vis)) # don't forget actual videos can't be b/w 0 and 1
		else:
			out.write(vis)
			#out.write((vis).astype('u1')) # should work for UN-NORMALIZED frames
		
	out.release()
	if play_vid == 1:
		cv2.destroyAllWindows()
		
	return 	
	
####################################################################################################################################	
################################################# V-Rep Simulation Specific ########################################################
####################################################################################################################################	

def GetFoldersForRuns():
	folders = []
	for x in os.listdir(os.getcwd()):
		if x.startswith('u_sequence'):
			folders.append(x)
	
	return folders
	
def LoadFramesActionsFromFolder(folder, CAM_W, CAM_H, CAM_C, ACTION_LEN): # note: by CAM_C we mean 3 rgb channels x # of cameras (e.g. 3*4 = 12)

	curr_dir = os.getcwd()
	search_dir = ""
	if platform == "win32":
		search_dir = os.getcwd()+"\\"+folder # WINDOWS
	else:
		search_dir = os.getcwd()+"/"+folder # LINUX
	os.chdir(search_dir)
	files = filter(os.path.isfile, os.listdir(search_dir))
	files = [os.path.join(search_dir, f) for f in files] # add path to each file
	files.sort(key=lambda x: x) #os.path.getmtime(x))
	os.chdir(curr_dir)
	
	timesteps = int(len(files) / 2)
	#print(len(files)); print(timesteps)
	frames = np.zeros((timesteps, CAM_W, CAM_H, CAM_C), dtype=np.uint8)	# IMPORTANT!! Or else value copy fails.. 
	actions = np.zeros((timesteps, ACTION_LEN)) # default float type should be fine
	i1 = 0
	i2 = 0
	
	for f in files:
		if f.endswith('_x.npy'):		# frame
			#print(f)
			temp = np.load(f)
			frames[i1,:,:,:] = temp[:,:,:]
			i1 = i1 + 1
		elif f.endswith('_u.npy'):		# action
			#print(f)
			actions[i2,:] = np.load(f).reshape(ACTION_LEN)
			i2 = i2 + 1
		else:
			print("ERROR: found file in %s not ending with _x.npy or _u.npy"%folder)
			print("File: %s"%f)
			return
			
	frames = NormalizeImgTensor(frames) # normalize
	
	return frames, actions	
	
def GetFrame4(frame, isVertical): 
### NOTE: imshow does weird things if input is float:
# "If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255]."

	img1 = frame[:,:,0:3]
	img2 = frame[:,:,3:6]
	img3 = frame[:,:,6:9]
	img4 = frame[:,:,9:12]
	if (isVertical == 1):
		img = np.concatenate((np.concatenate((img1, img2),axis=0), np.concatenate((img3, img4),axis=0)), axis=0)
	else:
		img = np.concatenate((np.concatenate((img1, img2),axis=1), np.concatenate((img3, img4),axis=1)), axis=0)
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	return img
	
### TODO / under construction	
def ViewFutureFrames(frames): # assumes frames of shape (NFF, CAM_W, CAM_H, CAM_C)
	num_frames = frames.shape[0] #
	
	font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
	
	imgs = GetFrame4(frames[0,:], 1)
	#cv2.putText(imgs,'OpenCV',(5,20), font, 1,(255,255,255),1,cv2.LINE_AA)
	for t in range(1, num_frames):
		next_img = GetFrame4(frames[t,:], 1)
		#cv2.putText(next_img,'[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]',(5,10), font, 0.2,(20,255,20),1,cv2.LINE_AA)
		#cv2.putText(next_img,'[0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71]',(5,25), font, 0.2,(255,20,20),1,cv2.LINE_AA)
		imgs = np.concatenate((imgs, next_img), axis=1)
		
	return imgs