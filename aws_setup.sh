#!/bin/bash

############################################
############ SETUP SCRIPT ##################
############################################

# Working with AWS instance, tested on Ubuntu 14.04:

# PROTIP: choose an older UBUNTU instance, e.g. 14.04
# PERFORMANCE COMPARISON:
# CLSTM:
# 	- my machine: ~1 min / batch (of ~15 images)
#	- amazon (crap) CPU: ~1 secs / batch
#	- amazon good CPU (16 cores, 64GB ram): ~4.5s / batch (15x speedup!!)
#	- amazon GPU: 

# PROTIP: using tmux
# ssh into the remote machine
# start tmux by typing tmux into the shell
# start the process you want inside the started tmux session
# leave/detach the tmux session by typing Ctrl+B and then D
# (can reattach: tmux attach)

# PROTIP: accessing Tensorboard from remote machine
# 1. Modify ssh login to use -L option:
# ssh -i "keything.pem" -L 16006:127.0.0.1:6006 user@HOSTNAME 
# Now your machine's 16006 port is the remote machine's 6006 port. 
# Now you can e.g. use tmux to start tensorboard

# Random Protips: 
# ps -ef             # view processes
# kill -9 [pid] 	 # kill process
# du -sh [dir]		 # folder/file size
# tail -f [file]	 # read the last 10 lines to terminal 

# general..
sudo apt-get update
sudo apt-get install git build-essential

# for ANACONDA3: 
cd ~
sudo wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
sudo bash Anaconda3-4.2.0-Linux-x86_64.sh
export PATH=/home/ubuntu/anaconda3/bin:$PATH # or where you chose to install it, will show up during install instructions above

# Tensorflow, OpenCV: 
sudo `which pip` install tensorflow
sudo apt-get install libgtk2.0-0
sudo `which conda` install -c menpo opencv3

# own code
cd ~
git clone https://github.com/js117/AI_Software.git




############# OLD / OTHER ###########
## 1. On log-in, install everything we need. In particular:
## use "sudo yum install [stuff]" if on redhat
#sudo apt-get update
#sudo apt-get install git build-essential
#sudo apt-get install python-setuptools python-dev 
#sudo apt-get install python-tk 
#sudo apt-get install cmake
#sudo easy_install pip
#sudo pip install tensorflow # sudo `which pip` install tensorflow
#sudo pip install matplotlib

# for openCV.. 
#git clone https://github.com/opencv/opencv.git
#cd ~/opencv
#mkdir release
#cd release
#sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
#sudo make
#sudo make install




