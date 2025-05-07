import numpy as np
import spiceypy as spice

from helpers import kepler_to_cartesian
from astrogator import Segment

class Initial_State(Segment):
	"""
	Class for defining the initial state of an Astrogator object
	"""

	def __init__(self, initial_state:np.ndarray, initial_time:str, mu:float, coord_type:str="keplerian"):
		self.initial_et = spice.str2et(initial_time)
		if coord_type.casefold() == "keplerian":
			self.initial_state = kepler_to_cartesian(initial_state, self.initial_et, mu)
		elif coord_type.casefold() == "cartesian":
			self.initial_state = initial_state
		else:
			raise TypeError("Error: Please specify a valid coordinate system type.")
	
	def run(self):
		return self.initial_et, self.initial_state