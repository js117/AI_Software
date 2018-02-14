# Video Next Frame Prediction with Convolutional LSTMs

AI_Software - General directory organization:

* Please run with Python 3.5, and latest Tensorflow version.
- Each project is self-contained in its respective folder, designed to run on a remote (Linux) machine. 
- Each project folder has a main (e.g. python) script to run. 
- Machines generally need to be setup with various installs and dependencies: try running "sh aws.setup.sh"

- Run procedure: "python <main_script.py> <prev_model_checkpoint.meta> <prev_model_num_steps>" 
	* prev_model_* arguments are optional, and correspond to loading and continuing training on previous runs. 

Using tmux: 

- ssh into the remote machine
- start tmux by typing tmux into the shell
- start the process you want inside the started tmux session
- leave/detach the tmux session by typing Ctrl+B and then D