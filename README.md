# AI Software

Some basic setup tips on Windows: 

- Start with the latest Anaconda
			https://www.continuum.io/downloads
			
- opencv: 
			conda install python=3.5
			conda install -c menpo opencv3
			
- ML libraries: 
			conda install tensorflow
			pip install keras
			
- Kinect SDK:
			https://www.microsoft.com/en-ca/download/details.aspx?id=44561 
			pip install pykinect2 
			(^ possible fix needed for 64-bit version: https://github.com/Kinect/PyKinect2/issues/37) 
			
			
Some basic setup tips on Ubuntu: 

	# Notepadqq, useful text editor
	sudo add-apt-repository ppa:notepadqq-team/notepadqq
	sudo apt-get update
	sudo apt-get install notepadqq
	
	# Nvidia drivers: 
	# https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/tensorflow/ 
	