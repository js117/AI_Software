import sys
import time
import math
import numpy as np
from numpy import linalg as LA
import time

def SolveSubproblem2(twist1, twist2, p, q, r):

    t11 = 0; t21 = 0; t12 = 0; t22 = 0;

    tol = 1e-6;
    w1 = twist1[3:6];
    w2 = twist2[3:6];
    
    r = r[0:3];
    u = p[0:3] - r;
    v = q[0:3] - r;
    
    # find z, the axis from intersection of w1/w2 to their common point(s)
    # of rotation
    # z = alpha*w1 + beta*w2 + gamma*cross(w1, w2)
    
    alpha = ((w1.T.dot(w2))*w2.T.dot(u) - w1.T.dot(v)) / ((w1.T.dot(w2))**2 - 1);
    
    beta = ((w1.T.dot(w2))*w1.T.dot(v) - w2.T.dot(u)) / ((w1.T.dot(w2))**2 - 1);
    
    # solve for gamma param: 
    temp = (u.T.dot(u) - alpha*alpha - beta*beta - 2*alpha*beta*(w1.T.dot(w2))) / LA.norm(cross(w1,w2),2)**2;
    
    if (temp < 0): # the fudge: assume all targets are feasible. If they are not we deal with it upstream, and it's the user's problem.
        temp = 0;

    gamma = temp**0.5;
    
    if (gamma < tol): # gamma ~ 0, so single solution
        z = alpha*w1 + beta*w2;
        c = z + r;
        t21 = SolveSubproblem1(twist2, p, c, r);
        t11 = SolveSubproblem1(-twist1, q, c, r);
        
        # copy to second possible solution
        t22 = t21;
        t12 = t11;  
    else:
        z1 = alpha*w1 + beta*w2 + gamma*cross(w1, w2);
        z2 = alpha*w1 + beta*w2 - gamma*cross(w1, w2);
        c1 = z1 + r;
        c2 = z2 + r;
        
        t21 = SolveSubproblem1(twist2, p, c1, r);
        t11 = SolveSubproblem1(-twist1, q, c1, r);
        
        t22 = SolveSubproblem1(twist2, p, c2, r);
        t12 = SolveSubproblem1(-twist1, q, c2, r);       
    
    return np.array([t11,t21,t12,t22]).reshape(4,1)


# Requires p,q,r to be of shape (4,1), twist is of shape (6,1)
def SolveSubproblem1(twist, p, q, r):

    w = twist[3:6];
    u = p[0:3] - r[0:3];
    v = q[0:3] - r[0:3];

    u_prime = u - w*(w.T.dot(u));
    v_prime = v - w*(w.T.dot(v));
    
    t1 = atan2(w.T.dot(cross(u_prime, v_prime)), u_prime.T.dot(v_prime));
    return t1


# Returns the yaw / pitch / roll (ZYX) Euler angles as a homogeneous transform
def ZYXEulerMatrix(zAngle,yAngle,xAngle):
    Rx = np.array([[1,0,0], [0,cos(xAngle),-sin(xAngle)], [0,sin(xAngle),cos(xAngle)]]);
    
    Ry = np.array([[cos(yAngle),0,sin(yAngle)], [0,1,0], [-sin(yAngle),0,cos(yAngle)]]);
    
    Rz = np.array([[cos(zAngle),-sin(zAngle),0], [sin(zAngle),cos(zAngle),0], [0,0,1]]);

    Reuler = Rz.dot(Ry).dot(Rx)
    
    final = np.zeros((4,4))
    final[0:3,0:3] = Reuler
    final[3,3] = 1
    
    return final
    
# Uses Rodriguez' formula for e^(w_hat*t), assuming norm(w) == 1, where hat is the wedge operator
def exp_twist_theta_revolute(e,t):
    
    v = e[0:3];
    w = e[3:6];
    
    w_hat = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]]);
    
    expWt = np.identity(3) + w_hat*sin(t) + w_hat.dot(w_hat)*(1 - cos(t));

    M = np.zeros((4,4))
    M[0:3,0:3] = expWt
    M[0:3,3] = (np.identity(3) - expWt).dot(cross(w,v)).reshape(3) + (w*(w.T.dot(v))*t).reshape(3)
    M[3,3] = 1

    return M
	
def GetRfromH(H):
	return H[0:3,0:3]

def cross(x,y):
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]]); #skew symmetric matrix representation
    return x_hat.dot(y)
    
def cos(x):
    return np.cos(x)
    
def sin(x):
    return np.sin(x)
    
def atan2(y,x):
    return np.arctan2(y,x)
	
global w1,w2,w3,w4,w5,w6
global q1,q2,q3,q4,q5,q6
global x1,x2,x3,x4,x5,x6

# CAESAR PARAMS -- DEC 2017 #
w1 = np.array([0,0,1])
w2 = np.array([0,1,0])
w3 = np.array([0,1,0])
w4 = np.array([1,0,0])
w5 = np.array([0,1,0])
w6 = np.array([1,0,0])

q1 = np.array([0,0,0])
q2 = np.array([0.18,0.18,0])
q3 = np.array([0.395,0.09,0])
q4 = np.array([0.45,0.055,0])
q5 = np.array([0.66,0,0])
q6 = np.array([0.74,-0.07,0])
	
x1 = np.array([0.03,0.11,0.41])	
x2 = np.array([0.435,0.15,0.11])	
x3 = np.array([0.36,0.07,0])	
x4 = np.array([0.57,0.12,0.05])	
x5 = np.array([0.63,-0.135,0])	
x6 = np.array([0.82,-0.09,0])	
xe = np.array([0.89,-0.075,-0.05])

e1 = np.concatenate((cross(-1*w1, q1), w1), axis=0)
e2 = np.concatenate((cross(-1*w2, q2), w2), axis=0)
e3 = np.concatenate((cross(-1*w3, q3), w3), axis=0)
e4 = np.concatenate((cross(-1*w4, q4), w4), axis=0)
e5 = np.concatenate((cross(-1*w5, q5), w5), axis=0)
e6 = np.concatenate((cross(-1*w6, q6), w6), axis=0)

# Initial IMU data from reference position (robot facing WEST in Oliver's garage)
# Format: [ID, roll, pitch, yaw, ax, ay, az] 
s4_rpy_axayaz = np.array([4, -2.64, -0.06, 45.03, 0.0, -0.05, 0.95])		# (theta1) SHOULDER YAW:		ID#4
s3_rpy_axayaz = np.array([3, 0.07, 0.04, 82.09, 0.0, 0.0, 1.0])				# (theta2) SHOULDER PITCH: 		ID#3
s6_rpy_axayaz = np.array([6, -1.38, 0.13, 50.56, 0.0, -0.02, 0.97])			# (theta5) WRIST PITCH			ID#6
s1_rpy_axayaz = np.array([1, -179.92, 0.12, -56.89, 0.0, 0.0, -1.0])		# (theta6) END EFFECTOR: 		ID#1
s2_rpy_axayaz = np.array([2, 0.01, 0.01, -165.41, 0.0, 0.0, 0.97])			# (theta4) ELBOW ROLL:   		ID#2
s5_rpy_axayaz = np.array([5, -179.41, 0.03, -77.59, 0.0, -0.01, -1.01])		# (theta3) ELBOW PITCH:	 		ID#5
	
#################################

if __name__ == "__main__":
	print("Testing kinematics program: ")
	
	
	print(w1)
	print(-1*w1)
	print(w1.shape)
	print("-----")
	print(q1)
	print(q1.shape)
	
	
	print(e1); print("---")
	print(e2); print("---")
	print(e3); print("---")
	print(e4); print("---")
	print(e5); print("---")
	print(e6); print("---")
	
	test1 =  exp_twist_theta_revolute(e1,t=0)
	print(test1); print("---")
	test2 =  exp_twist_theta_revolute(e2,t=0)
	print(test2); print("---")
	test3 =  exp_twist_theta_revolute(e3,t=0)
	print(test3); print("---")
	test4 =  exp_twist_theta_revolute(e4,t=0)
	print(test4); print("---")
	test5 =  exp_twist_theta_revolute(e5,t=0)
	print(test5); print("---")
	test6 =  exp_twist_theta_revolute(e6,t=0)
	print(test6); print("---")
	
	
	
	
	
	
	
	