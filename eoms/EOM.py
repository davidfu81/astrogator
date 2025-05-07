import numpy as np
from typing import Protocol

class EOM(Protocol):
	"""
	Abstract class for EOM classes to implement
	"""
	def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
		"""
		Computes the equations of motion.

		Parameters:
		t (float)
			Time variable
		state (array-like): 
			State vector [x, y, z, vx, vy, vz].
		
		Returns:
		np.ndarray: Time derivatives [vx, vy, vz, ax, ay, az].
		"""
		pass