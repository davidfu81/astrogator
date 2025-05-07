import numpy as np

from astrogator import Segment
from integrators import Integrator
from eoms import EOM

class Propagate(Segment):
	def __init__(self, duration:float, integrator:Integrator):
		self.duration = duration
		self.integrator = integrator

	def run(self, t0:float, state0:np.ndarray)->tuple[np.ndarray, np.ndarray]:
		"""
		Run the propagation
		
		Parameters:
		t0 : float
			Initial time (in ephemeris seconds) of the propagation
		state0 : np.ndarray
			(1,6) array that contains the initial Cartesian state of the propagation
		
		Returns:
		t : np.ndarray
			(N,) Time array of the propagated ephemeris 
		state : np.ndarray
			(N,6) State array of the propagated ephemeris
		"""
		states, time = self.integrator.integrate(state0, t0, t0+self.duration, step=60)
		return time, states