import numpy as np

from eoms import EOM 

class RKDP54:
	"""
	This class implements the Dormand Prince method of order 5 (4) for solving
	an initial value problem for ODEs.
	"""
	def __init__(self, eom:EOM, abs_tol=1e-3, rel_tol = 1e-3):
		self.eom = eom
		self.abs_tol = abs_tol
		self.rel_tol = rel_tol

	def integrate(self, y0:np.ndarray, t0:float, tf:float, step:float)->tuple[np.ndarray, np.ndarray]:
		h = step	# Default step size
		tout = np.arange(t0, tf, step)
		if tf not in tout:
			tout = np.concatenate((tout, [tf]))
		yout = np.zeros((len(tout),6))
		yout[0] = y0
		curr_i = 1
		prev_err = 1e-4
		y = y0
		t = t0
		dydx = self.eom.__call__(t0, y0)

		#Dormand Prince Coefficients for 5(4) Embedded Runge-Kutta
		c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
		a = np.array([
			[0, 0, 0, 0, 0, 0, 0],
			[1/5, 0, 0, 0, 0, 0, 0],
			[3/40, 9/40, 0, 0, 0, 0, 0],
			[44/45, -56/15, 32/9, 0, 0, 0, 0],
			[19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
			[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
			[35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
		])
		k = np.zeros((7,len(y)))
		# Error estimation
		e = np.array([71/57600, 0, -71/16695, 1/1920, -17253/339200, 22/525, -1.0/40])
		# Dense Output
		d = np.array([-12715105075/11282082432, 0, 87487479700/32700410799, -10690763975/1880347072,
				701980252875/199316789632, -1453857185/822651844, 69997945/29380423])
		r = np.zeros((5,len(y0)))

		# Step Controller Parameters
		order = 5
		alpha = 0.7/order
		beta = 0.4/order
		reject = False
		safety = 0.95
		max_factor = 10
		min_factor = 5

		while t < tf:
			# Calculate integration step with adaptive step size control
			while True:
				if np.isnan(h):
					return
				# Evaluate 5(4) step
	
				k[0] = h*dydx
				for i in range(1,6):
					k[i] = h*self.eom.__call__(t+c[i]*h, y+np.dot(a[i,:],k))
				ynew = y+np.dot(a[6,:],k)
				dydxnew = self.eom.__call__(t+h, ynew)
				k[6] = h*dydxnew

				# Compute error
				delta = np.dot(e,k)
				scale = self.abs_tol+self.rel_tol*np.maximum(np.abs(ynew), np.abs(y))
				err = np.sqrt(1/6*np.sum((delta/scale)**2))

				# Step size control
				if err <= 1:
					if err == 0:
						factor = max_factor
					factor = np.clip(safety*err**(-alpha)*prev_err**(-beta),1/min_factor, max_factor)
					if reject:
						factor = min(1/min_factor, factor)
					prev_err = max(err, 1e-4)
					reject = False
				else:
					factor = max(safety*err**(-alpha), 1/min_factor)
					reject = True
				h *= factor
				if not reject:
					break

			# Prepare coefficients for dense output
			if curr_i < len(tout) and tout[curr_i] <= t+h:
				r[0] = y
				r[1] = ynew-y
				r[2] = k[0]-r[1]
				r[3] = r[1]-k[6]-r[2]
				r[4] = np.dot(d,k)
			
				# Step Forward with dense output
				i_start = curr_i
				i_end = np.searchsorted(tout, t + h)
				thetas = (tout[i_start:i_end] - t) / h
				thetas_col = thetas[:, None]
				yout[i_start:i_end] = r[0] + thetas_col * (r[1] + (1 - thetas_col) * (
					r[2] + thetas_col * (r[3] + (1 - thetas_col) * r[4])))
				curr_i = i_end
				# while curr_i < len(tout) and tout[curr_i] <= t+h:
				# 	theta = (tout[curr_i]-t)/h
				# 	yout[curr_i] = r[0]+theta*(r[1]+(1-theta)*(r[2]+theta*(r[3]+(1-theta)*r[4])))
				# 	curr_i += 1
			y = ynew
			t = t+h
			dydx = dydxnew

		return yout, tout