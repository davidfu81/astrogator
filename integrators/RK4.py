import numpy as np

from eoms import EOM 

class RK4:
	"""
	This class implements the Dormand Prince method of order 5 (4) for solving
	an initial value problem for ODEs.
	"""
	def __init__(self, eom:EOM):
		self.eom = eom

	def integrate(self, y0:np.ndarray, t0:float, tf:float, step:float)->tuple[np.ndarray, np.ndarray]:
		time = np.arange(t0, tf, step)
		if tf not in time:
			time = np.concatenate((time, [tf]))
		states = np.zeros((len(time),6))
		states[0] = y0

		for i, (ti, yi) in enumerate(zip(time[:-1], states[:-1])):
			h = time[i+1]-ti
			k1 = h*self.eom.__call__(ti, yi)
			k2 = h*self.eom.__call__(ti+h/2, yi+k1/2)
			k3 = h*self.eom.__call__(ti+h/2, yi+k2/2)
			k4 = h*self.eom.__call__(ti+h, yi+k3)

			states[i+1] = yi+k1/6+k2/3+k3/3+k4/6
		
		return states, time