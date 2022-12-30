import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import activations as act
import numpy as np
import math

def sigmoid(a,b) :
	x = np.linspace(a,b,20)
	y = act.sigmoid(x)
	x_y_spline = make_interp_spline(x,y)

	xnew = np.linspace(a,b,300)
	ynew = x_y_spline(xnew)

	plt.plot(xnew,ynew)
	plt.show()

def tanh(a,b) :
	x = np.linspace(a,b,20)
	y = act.tanh(x)
	x_y_spline = make_interp_spline(x,y)

	xnew = np.linspace(a,b,300)
	ynew = x_y_spline(xnew)

	plt.plot(xnew,ynew)
	plt.show()

def ReLU(a,b) :
	if (a%1==0 and b%1==0):
		x = np.arange(a,b+1)
		xnew = np.arange(a,b+1)
	else :
		a_ = np.array([a])
		b_ = np.array([b])
		x_ = np.arange(math.ceil(a),math.floor(b)+1)
		x  = np.concatenate((a_,x_,b_),axis=None)
		xnew = np.concatenate((a_,x_,b_),axis=None)
	y = act.ReLU(xnew)
	plt.plot(x,y)
	plt.show()
