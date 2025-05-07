import numpy as np

from astrogator import Segment

class ImpusliveBurn(Segment):
	def __init__(self, dv, burn_dir:str="along_vel"):
		self.dv_vec = dv
		self.burn_dir = burn_dir

	def run(self, t0:float, state0:np.ndarray)->tuple[np.ndarray, np.ndarray]:
		"""
		Run the finite burn
		
		Parameters:
		t0 : float
			Initial time (in ephemeris seconds) of the propagation
		state0 : np.ndarray
			(1,6) array that contains the initial Cartesian state of the propagation
		
		Returns:
		t : np.ndarray
			(1,) Time array of the burn
		state : np.ndarray
			(1,6) State array of the burn
		"""
		burn_with_state = state0.copy()
		if self.burn_dir == "along_vel":
			burn_with_state[0,3:] += self.dv*burn_with_state[0,3:]/np.linalg.norm(burn_with_state[0,3:])
		return t0, burn_with_state