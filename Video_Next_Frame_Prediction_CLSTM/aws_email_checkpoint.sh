#!/bin/bash

############################################
############ BACKUP SCRIPT #################
############################################

# PWD: inside the main folder of an AI project. Should contain the main running script.
# Also in this main project folder, there will be checkpoints, and a logs/ folder. 

# basic email usage with mutt: 
# echo "This is text in the body." | mutt -a fille_to_attach.zip -s "Email subject" -c <email of recipient>

# Increasing limit:
# sudo nano /etc/postfix/main.cf
# message_size_limit = [new number] # (default: 10240000 = 10 MB)

# Tip when commiting from remote machine, don't include big logs folder:
# git add -A 
# git reset -- somefolder_to_exclude