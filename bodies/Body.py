import numpy as np
import spiceypy as spice

class Body():
	"""
	Class to store information about a simulation's central body

	Paramters
	mu : float
		Gravitational Parameter (km**3/s**2)
	radius : float
		Equatorial radius (km)
	"""
	def __init__(self, mu:float, radius:float):
		self.mu = mu
		self.radius = radius
	
	def cartesian_to_kepler(self, states:np.ndarray, time:np.ndarray):
		"""
		Converts (N,6) Cartesian state vectors to (N,6) Keplerian orbital elements.

		Parameters:
		states: np.ndarray
			(N,6) Cartesian states (pos, vel) in km, km/s
		time: np.ndarray
			(N,1) time steps in ephemeris time
		
		Returns:
		np.ndarray
			-(N,6) Keplerian elements where each column is:
			[0] - semimajor axis (a) [km]
			[1] - eccentricity (e) [-]
			[2] - inclination (i) [rad]
			[3] - RAAN (Omega) [rad]
			[4] - argument of periapse (omega) [rad]
			[5] - true anomaly (nu) [rad]
		"""

		element_indices = np.r_[0:6, 8:11]
		if states.ndim == 1:
			elements = spice.oscltx(np.ascontiguousarray(states), time, self.mu)[element_indices]
		else:
			elements = np.zeros([len(states), 9])
			for i, (ti, si) in enumerate(zip(time, states)):
				element = spice.oscltx(np.ascontiguousarray(si), ti, self.mu)[element_indices]
				elements[i, :] = element

		return elements[:,(7,1,2,3,4,6)]
	
	def kepler_to_cartesian(self, elts:np.ndarray, time:np.ndarray):
		"""
		Converts (N,6) Keplerian orbital elements to (N,6) Cartesian State Vector.

		Parameters
		elts: np.ndarray
			(N,6) Keplerian elements where each column is:
			[0] - semimajor axis (a) [km]
			[1] - eccentricity (e) [-]
			[2] - inclination (i) [rad]
			[3] - RAAN (Omega) [rad]
			[4] - argument of periapse (omega) [rad]
			[5] - true anomaly (nu) [rad]
		time: np.ndarray
			(N,1) time steps in ephemeris time

		Output
		np.ndarray
			- (N,6) Cartesian states (pos, vel) in km, km/s
		"""

		# Convert to Cartesian
		if elts.ndim == 1:
			a, e, i, raan, argp, true_anom = elts

			rp = a*(1-e)
			
			# Mean anomaly
			eccen_anom = np.arctan2(np.sqrt(1 - e**2) * np.sin(true_anom),
								1 + e*np.cos(true_anom))
			mean_anom = eccen_anom - e*np.sin(eccen_anom)
			mean_anom = np.mod(mean_anom, 2*np.pi)

			# Build SPICE-compliant orbital elements array: (N,8)
			oscelt = np.array([rp, e, i, raan, argp, mean_anom, time, self.mu])
			states = spice.conics(oscelt, time)
		else:
			states = np.zeros_like(oscelt)
			for i, (ti, ei) in enumerate(zip(time, elts)):
				a, e, i, raan, argp, true_anom = ei[0]
				rp = a*(1-e)
				
				# Mean anomaly
				eccen_anom = np.arctan2(np.sqrt(1 - e**2) * np.sin(true_anom),
									1 + e*np.cos(true_anom))
				mean_anom = eccen_anom - e*np.sin(eccen_anom)
				mean_anom = np.mod(mean_anom, 2*np.pi)

				# Build SPICE-compliant orbital elements array: (N,8)
				oscelt = np.array([rp, e, i, raan, argp, mean_anom, ti, self.mu])
				state = spice.conics(oscelt, ti)
				states[i, :] = state

		return states

	def convert_states_to_oscelt (self, time, states):
		"""
		Convert state vectors to SPICE compliant orbital elements.

		This function converts state vectors (position and velocity) to orbital elements using SPICE's oscelt function. 

		Parameters:
		- time (float or array-like): The time or times at which the state vectors are defined. 
									If `states` is an array, `time` should be an array of the same length.
		- states (array-like): The state vector or vectors to be converted. 
							Each state vector should be in the form [x, y, z, dx, dy, dz].
							This can be a 1D array (for a single state vector) or a 2D array (for multiple state vectors).

		Returns:
		- elements (array-like): The converted orbital elements. 
								Each element is in the form [rp, e, i, Ω, w, M, t0, mu], where:
								rp - Radius of perigee
								e - Eccentricity
								i - Inclination                                
								Ω - Longitude of the ascending node
								w - Argument of periapsis
								M - Mean anomaly
								t0 - epoch
								mu - gravitational parameter
								For a single state vector, this is a 1D array. For multiple state vectors, this is a 2D array.

		"""
		if states.ndim == 1:
			elements = spice.oscelt(np.ascontiguousarray(states), time, self.mu)
		else:
			elements = np.zeros([len(states), 8])
			for i, (ti, si) in enumerate(zip(time, states)):
				element = spice.oscelt(np.ascontiguousarray(si), ti, self.mu)
				elements[i, :] = element

		return elements

	def convert_oscelt_to_states(self, time, oscelt):
		"""
		Convert orbital elements to state vectors.

		This function converts orbital elements to state vectors (position and velocity) using SPICE's conics function. 
		The conversion is done for either a single set of orbital elements or an array of orbital elements.

		Parameters:
		- time (float or array-like): The time or times at which the orbital elements are defined. 
									If `elements` is an array, `time` should be an array of the same length.
		- elements (array-like): The orbital elements or sets of elements to be converted. 
								Each element is in the form [rp, e, i, Ω, w, M, t0, mu], where:
								rp - Radius of perigee
								e - Eccentricity
								i - Inclination                                
								Ω - Longitude of the ascending node
								w - Argument of periapsis
								M - Mean anomaly
								t0 - epoch
								mu - gravitational parameter
								For a single state vector, this is a 1D array. For multiple state vectors, this is a 2D array.

		Returns:
		- states (array-like): The converted state vectors. 
							Each state vector is in the form [x, y, z, dx, dy, dz].
							For a single set of orbital elements, this is a 1D array. For multiple sets, this is a 2D array.

		Note:
		The function calculates the state vector based on the provided orbital elements and time.
		"""
		
		if oscelt.ndim == 1:
			states = spice.conics(oscelt, time)
		else:
			states = np.zeros_like(oscelt)
			for i, (ti, ei) in enumerate(zip(time, oscelt)):
				state = spice.conics(ei, ti)
				states[i, :] = state

		return states
	
	def convert_sma_to_period(self, sma):
		"""
		Find the orbital period [s] from the semi-major axis [km] (and mu [km^3/s^2])

		Parameters:
			semi-major axis (numpy.ndarray): Array of semi-major axis.

		Returns:
			numpy.ndarray: Array of orbital periods
		"""

		return 2 * np.pi * np.sqrt(sma**3 / self.mu)