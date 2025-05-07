import numpy as np

class Two_Body_EOM():
	def __init__(self, mu:float):
		"""
		Two_Body_EOM constructor

		Paramters:
		mu : float
			Gravitational parameter (km**3/s**2)
		"""
		self.mu = mu
	
	def __call__(self, t:float, state):
		"""
		Computes the two-body equations of motion.

		Parameters:
		t : float
			Time variable, not used here as the system is time-invariant, but required for solvers.
		state : array-like 
			State vector [x, y, z, vx, vy, vz]. (km,km/s)
		
		Returns:
		np.ndarray: Time derivatives [vx, vy, vz, ax, ay, az].
		"""
		# Unpack position and velocity
		x, y, z, vx, vy, vz = state
		
		# Compute the distance from the central body
		r = np.sqrt(x**2 + y**2 + z**2)
		
		# Calculate accelerations due to gravity
		ax = -self.mu * x / r**3
		ay = -self.mu * y / r**3
		az = -self.mu * z / r**3
		
		# Return the derivatives of position and velocity
		return np.array([vx, vy, vz, ax, ay, az]) 