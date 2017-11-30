import cv2
import numpy as np
import time
import os
import serial
import sys
import threading
import msvcrt # WINDOWS ONLY (what does Linux need?)
import pickle
import array as ARR
from collections import deque
#import pylab
#from pylab import *
#import tkinter as Tkinter
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib
from matplotlib import pyplot as plt
from pylab import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

global IMUSerialControllers
global NumIMUs
global baudrate
global BytesPerBuffer

global plot_time_axes
global IMU_BUFFER_SIZE
global IMU_buffer_roll
global IMU_buffer_pitch
global IMU_buffer_yaw
global IMU_buffer_ax
global IMU_buffer_ay
global IMU_buffer_az
global IMU_buffer_gx
global IMU_buffer_gy
global IMU_buffer_gz
global IMU_buffer_mx
global IMU_buffer_my
global IMU_buffer_mz

### Some initializations: 
baudrate = 115200
NumIMUs = 2
BytesPerBuffer = 13*4 +2		# 13 float values being sent via Arduino, 4 bytes each; plus newline and carriage return (see Arduino code)
IMUSerialControllers = []

IMU_BUFFER_SIZE = 300
plot_time_axes = [i for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_roll = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)] 
IMU_buffer_pitch = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_yaw = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_ax = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_ay = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_az = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gx = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gy = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_gz = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_mx = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_my = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
IMU_buffer_mz = [deque([0] * IMU_BUFFER_SIZE) for i in range(0,IMU_BUFFER_SIZE)]
# d.append(newElem); d.popleft()


def serial_ports():
	""" Lists serial port names

		:raises EnvironmentError:
			On unsupported or unknown platforms
		:returns:
			A list of the serial ports available on the system
	"""
	if sys.platform.startswith('win'):
		ports = ['COM%s' % (i + 1) for i in range(256)]
	elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
		# this excludes your current terminal "/dev/tty"
		ports = glob.glob('/dev/tty[A-Za-z]*')
	elif sys.platform.startswith('darwin'):
		ports = glob.glob('/dev/tty.*')
	else:
		raise EnvironmentError('Unsupported platform')

	result = []
	for port in ports:
		try:
			s = serial.Serial(port)
			s.close()
			result.append(port)
		except (OSError, serial.SerialException):
			pass
	return result
	
	
ports = serial_ports()
print(ports)

'''
					WHAT WE GET FROM THE ARDUINOS:
					
  SEND_DATA[0] = dev_id;	// (unique and constant)
  SEND_DATA[1] = myIMU.roll;
  SEND_DATA[2] = myIMU.pitch;
  SEND_DATA[3] = myIMU.yaw;
  SEND_DATA[4] = (int)1000*myIMU.ax;
  SEND_DATA[5] = (int)1000*myIMU.ay;
  SEND_DATA[6] = (int)1000*myIMU.az;
  SEND_DATA[7] = myIMU.gx;
  SEND_DATA[8] = myIMU.gy;
  SEND_DATA[9] = myIMU.gz;
  SEND_DATA[10] = myIMU.mx;
  SEND_DATA[11] = myIMU.my;
  SEND_DATA[12] = myIMU.mz;
'''

for i in range(len(ports)):
	try:
		print("testing port: " + str(ports[i]))
		test_port = serial.Serial(ports[i], baudrate, timeout=3)
		
		bytes1 = test_port.read(BytesPerBuffer)
		bytes2 = test_port.read(BytesPerBuffer)
		bytes3 = test_port.read(BytesPerBuffer)
		lenb1 = len(bytes1)
		lenb2 = len(bytes2)
		lenb3 = len(bytes3)
		
		print("------------------")
		print(bytes1); print("--- len b1: " + str(lenb1))
		print(bytes2); print("--- len b2: " + str(lenb2))
		print(bytes3); print("--- len b3: " + str(lenb3))
		
		if lenb3 == BytesPerBuffer:
			print(str(ports[i]) + " is an IMU sensor port.")
			IMUSerialControllers.append(test_port)
			print("")
		
	except:
		print("failed to connect to port.")
		


	


###################################################### START MAIN PROGRAM ######################################################


class App(QtGui.QMainWindow):
	def __init__(self, parent=None):
		super(App, self).__init__(parent)

		#### Create Gui Elements ###########
		self.mainbox = QtGui.QWidget()
		self.setCentralWidget(self.mainbox)
		self.mainbox.setLayout(QtGui.QVBoxLayout())

		self.canvas = pg.GraphicsLayoutWidget()
		self.mainbox.layout().addWidget(self.canvas)

		self.label = QtGui.QLabel()
		self.mainbox.layout().addWidget(self.label)

		self.view = self.canvas.addViewBox()
		self.view.setAspectLocked(True)
		self.view.setRange(QtCore.QRectF(0,0, 100, 100))

		#  image plot
		#self.img = pg.ImageItem(border='w')
		#self.view.addItem(self.img)
		
		self.numIMUs = len(IMUSerialControllers)
		self.bufferLength = IMU_BUFFER_SIZE
		
		self.otherplot = [[self.canvas.addPlot(row=i,col=0, title="IMU #"+str(i)+" ROLL"),
						   self.canvas.addPlot(row=i,col=1, title="IMU #"+str(i)+" PITCH"),
						   self.canvas.addPlot(row=i,col=2, title="IMU #"+str(i)+" YAW")] 
						   for i in range(0,self.numIMUs)]
		self.h2 = [[self.otherplot[i][0].plot(pen='r'), self.otherplot[i][1].plot(pen='g'), self.otherplot[i][2].plot(pen='b')] for i in range(0,self.numIMUs)] 
		self.ydata = [[np.zeros((1,IMU_BUFFER_SIZE)),np.zeros((1,IMU_BUFFER_SIZE)),np.zeros((1,IMU_BUFFER_SIZE))] for i in range(0,self.numIMUs)]
		
		
		for i in range(0,self.numIMUs):
			self.otherplot[i][0].setYRange(min= -180, max= 180) 
			self.otherplot[i][1].setYRange(min= -180, max= 180) 
			self.otherplot[i][2].setYRange(min= -180, max= 180) 




		'''
		self.canvas.nextRow()
		#  line plot
		self.otherplot = self.canvas.addPlot()
		self.h2 = self.otherplot.plot(pen='y')
		
		self.canvas.nextRow()
		self.otherplot_2 = self.canvas.addPlot()
		self.h2_2 = self.otherplot_2.plot(pen='y')
		
		self.canvas.nextRow()
		self.otherplot_3 = self.canvas.addPlot()
		self.h2_3 = self.otherplot_3.plot(pen='y')
		
		self.canvas.nextRow()
		self.otherplot_4 = self.canvas.addPlot()
		self.h2_4 = self.otherplot_4.plot(pen='y')
		'''


		#### Set Data  #####################

		#self.x = np.linspace(0,50., num=100)
		#self.X,self.Y = np.meshgrid(self.x,self.x)

		self.counter = 0
		self.fps = 0.
		self.lastupdate = time.time()

		#### Start  #####################
		self._update()

	def _update(self):

		for i in range(0,self.numIMUs):
			self.ydata[i][0] = np.array(IMU_buffer_roll[i])
			self.ydata[i][1] = np.array(IMU_buffer_pitch[i])
			self.ydata[i][2] = np.array(IMU_buffer_yaw[i])
			
			self.h2[i][0].setData(self.ydata[i][0])
			self.h2[i][1].setData(self.ydata[i][1])
			self.h2[i][2].setData(self.ydata[i][2])
	
		'''
		self.data = np.sin(self.X/3.+self.counter/9.)*np.cos(self.Y/3.+self.counter/9.)
		self.ydata = np.sin(self.x/3.+ self.counter/9.)
		
		self.ydata_2 = np.sin(self.x/3.+ self.counter/9.)
		
		self.ydata_3 = np.sin(self.x/3.+ self.counter/9.)
		
		self.ydata_4 = np.sin(self.x/3.+ self.counter/9.)

		#self.img.setImage(self.data)
		
		self.h2.setData(self.ydata)
		
		self.h2_2.setData(self.ydata_2)
		
		self.h2_3.setData(self.ydata_3)
		
		self.h2_4.setData(self.ydata_4)
		'''
		
		now = time.time()
		dt = (now-self.lastupdate)
		if dt <= 0:
			dt = 0.000000000001
		fps2 = 1.0 / dt
		self.lastupdate = now
		self.fps = self.fps * 0.9 + fps2 * 0.1
		tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
		self.label.setText(tx)
		QtCore.QTimer.singleShot(1, self._update)
		self.counter += 1
		
def IMUGraphingThread():	
		
	app1 = QtGui.QApplication(sys.argv)
	thisapp1 = App()
	thisapp1.show()
	
	sys.exit(app1.exec_())


#IMUGraphingThread()	
	
	
imu_plotting_thread_1 = threading.Thread(target=IMUGraphingThread)
try:
	imu_plotting_thread_1.setDaemon(True) 
	imu_plotting_thread_1.start() 
except (KeyboardInterrupt, SystemExit):
	cleanup_stop_thread();
	sys.exit()	
		
	
# Test streaming data: 
while 1:
	#bytes = IMUSerialControllers[0].readline() #read(BytesPerBuffer)
	
	#bytes2 = IMUSerialControllers[1].readline()
	
	#print("----------------------")
	#print(bytes)
	#print("Len bytes: " + str(len(bytes)))
	
	for index,controller in enumerate(IMUSerialControllers):
	
		bytes = controller.readline()
	
		if (len(bytes) == BytesPerBuffer):
		
			imu_arr = ARR.array('f', bytes[:-2])		# KEY: subtract off the \r\n carriage return and newline final 2 bytes
			
			imu_id = imu_arr[0]
			imu_roll = imu_arr[1]
			imu_pitch = imu_arr[2]
			imu_yaw = imu_arr[3]
			imu_ax = imu_arr[4]
			imu_ay = imu_arr[5]
			imu_az = imu_arr[6]
			imu_gx = imu_arr[7]
			imu_gy = imu_arr[8]
			imu_gz = imu_arr[9]
			imu_mx = imu_arr[10]
			imu_my = imu_arr[11]
			imu_mz = imu_arr[12]
			
			IMU_buffer_roll[index].append(imu_roll); IMU_buffer_roll[index].popleft()
			IMU_buffer_pitch[index].append(imu_pitch); IMU_buffer_pitch[index].popleft()
			IMU_buffer_yaw[index].append(imu_yaw); IMU_buffer_yaw[index].popleft()
			IMU_buffer_ax[index].append(imu_ax); IMU_buffer_ax[index].popleft()
			IMU_buffer_ay[index].append(imu_ay); IMU_buffer_ay[index].popleft()
			IMU_buffer_az[index].append(imu_az); IMU_buffer_az[index].popleft()
			IMU_buffer_gx[index].append(imu_gx); IMU_buffer_gx[index].popleft()
			IMU_buffer_gy[index].append(imu_gy); IMU_buffer_gy[index].popleft()
			IMU_buffer_gz[index].append(imu_gz); IMU_buffer_gz[index].popleft()
			IMU_buffer_mx[index].append(imu_mx); IMU_buffer_mx[index].popleft()
			IMU_buffer_my[index].append(imu_my); IMU_buffer_my[index].popleft()
			IMU_buffer_mz[index].append(imu_mz); IMU_buffer_mz[index].popleft()

			
			print("------------------------------------------------------")
			print("id: " + str(imu_id))
			print("ROLL: " + str(imu_roll))
			print("PITCH: " + str(imu_pitch))
			print("YAW: " + str(imu_yaw))
			print("AX: " + str(imu_ax))
			print("AY: " + str(imu_ay))
			print("AZ: " + str(imu_az))
			print("GX: " + str(imu_gx))
			print("GY: " + str(imu_gy))
			print("GZ: " + str(imu_gz))
			print("MX: " + str(imu_mx))
			print("MY: " + str(imu_my))
			print("MZ: " + str(imu_mz))
		
			
