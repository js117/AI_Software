import sys
import threading

import win32api
import win32con

def T1Thread():
	while 1:
		print("1 --------------- thread 1 running")
		if win32api.GetAsyncKeyState(win32con.VK_F12) != 0:
			break
	sys.exit()
	
def T2Thread():
	while 1:
		print("2 --------------- thread 2 running")
		if win32api.GetAsyncKeyState(win32con.VK_F12) != 0:
			break
	sys.exit()
	
def T3Thread():
	while 1:
		print("3 --------------- thread 3 running")
		if win32api.GetAsyncKeyState(win32con.VK_F12) != 0:
			break
	sys.exit()
	
def T4Thread():
	while 1:
		print("4 --------------- thread 4 running")
		if win32api.GetAsyncKeyState(win32con.VK_F12) != 0:
			break
	sys.exit()


t1_thread = threading.Thread(target=T1Thread)
try:
	t1_thread.setDaemon(True)  # important for cleanup ? 
	t1_thread.start() # join too? 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()
print("---------- Started t1 thread. ----------")

t2_thread = threading.Thread(target=T2Thread)
try:
	t2_thread.setDaemon(True)  # important for cleanup ? 
	t2_thread.start() # join too? 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()
print("---------- Started t2 thread. ----------")

t3_thread = threading.Thread(target=T3Thread)
try:
	t3_thread.setDaemon(True)  # important for cleanup ? 
	t3_thread.start() # join too? 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()
print("---------- Started t3 thread. ----------")

t4_thread = threading.Thread(target=T4Thread)
try:
	t4_thread.setDaemon(True)  # important for cleanup ? 
	t4_thread.start() # join too? 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()
print("---------- Started t4 thread. ----------")

while 1:
	pass