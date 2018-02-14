import sys
import os 
import random
from random import randint
import numpy as np
import cv2
sys.path.append('..')
import Video_Utils
import CNN_Utils
import h5py
import time
import numpy as np

# make a video: 

vid_data = Video_Utils.GetDataFromFolder('aws_pics', do_normalize=0, w=1280, h=1054)
print(vid_data.shape)


Video_Utils.CreateOutputVideo(vid_data, output_file_name='learning_vis_2.avi', output_frame_rate=4, w=1280, h=1054, do_un_normalize=0, play_vid=1)