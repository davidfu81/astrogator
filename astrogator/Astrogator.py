import numpy as np
import spiceypy as spice

from astrogator import (
	Segment,
	Initial_State
)
from bodies import Body

class Astrogator():
	"""
	Class for running astrodynamics simulations
	"""
	def __init__(self, initial_state:np.ndarray, initial_time:str, cen_body:Body, coord_type:str="keplerian"):
		self.initial_et = spice.str2et(initial_time)

		if coord_type.casefold() == "keplerian":
			self.initial_state = cen_body.kepler_to_cartesian(initial_state, self.initial_et)
		elif coord_type.casefold() == "cartesian":
			self.initial_state = initial_state
		else:
			raise TypeError("Error: Please specify a valid coordinate system type.")
		
		self.cen_body = cen_body
		self.segments = []
	
	def add_segment(self, segment:Segment):
		"""
		Add an astrogator segment
		"""
		self.segments.append(segment)

	def run_all(self):
		"""
		Run all segments to generate an ephemeris
		"""
		self.time, self.states = np.array([self.initial_et]), self.initial_state.copy().reshape((-1,6))
		for segment in self.segments:
			# Run the segment
			assert type(segment) is not Initial_State, "There can only be one Initial State"
			seg_time, seg_states = segment.run(self.time[-1], self.states[-1])

			# Append results to time/state history
			self.time = np.hstack([self.time, seg_time])
			self.states = np.vstack([self.states, seg_states])
			