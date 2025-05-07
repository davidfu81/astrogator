import numpy as np

class Segment():
	"""
	This is a base class for Astrogator segments to inherit.
	"""
	def __init__(self):
		pass
	def run(self, t0:float, state0:np.ndarray)->tuple[np.ndarray, np.ndarray]:
		"""
		Run the segment
		
		Parameters:
		t0 : float
			Initial time (in ephemeris seconds) of the segment
		state0 : np.ndarray
			(1,6) array that contains the initial Cartesian state of the segment
		
		Returns:
		t : np.ndarray
			(N,) Time array of the segment ephemeris 
		state : np.ndarray
			(N,6) State array of the segment ephemeris
		"""
		pass