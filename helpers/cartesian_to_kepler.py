import numpy as np
import spiceypy as spice

def cartesian_to_kepler(states:np.ndarray, time:np.ndarray, mu:float):
	"""
	Converts (N,6) Cartesian state vectors to (N,6) Keplerian orbital elements.

	Parameters:
	states: np.ndarray
		(N,6) Cartesian states (pos, vel) in km, km/s
	time: np.ndarray
		(N,1) time steps in ephemeris time
	mu : float
		- graviational parameter of central body (km**3/s**2)
	
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
	# r_vec = state[:,:3]
	# v_vec = state[:,3:]
	# r_mag = np.linalg.norm(r_vec, axis=1)
	# v_mag = np.linalg.norm(v_vec, axis=1)

	# # Angular momentum vector
	# h_vec = np.cross(state[:,:3], state[:,3:])
	# h_mag = np.linalg.norm(h_vec, axis=1)
	# n_vec = np.cross(np.array([0, 0, 1]), h_vec)
	# n_mag = np.linalg.norm(n_vec, axis=1)

	# # Eccentricity vector
	# e_vec = ((v_mag**2-mu/r_mag)*r_vec - np.inner(r_vec,v_vec,axis=1)*v_vec)/mu
	# e_mag = np.linalg.norm(e_vec, axis=1)

	# # Semimajor axis
	# xi = v_mag**2/2-mu/r_mag
	# a = -mu/(2*xi)
	# a[e_mag == 1.0] = np.inf

	# # Inclination
	# i = np.arccos(h_vec[:,2]/h_mag)
	# # RAAN
	# raan = np.arccos(n_vec[:,0]/n_mag)
	# raan[n_vec[:,1]<0] = 2*np.pi - raan[n_vec[:,1]<0]
	# # Argument of periapse
	# arg_peri = np.arccos(np.inner(n_vec, e_vec,axis=1)/(n_mag*e_mag))
	# arg_peri[e_vec[:,2]<0] = 2*np.pi - arg_peri[e_vec[:,2]<0]
	# # True anomaly
	# nu = np.arccos(np.inner(e_vec,r_vec,axis=1)/(e_mag*r_mag))
	# nu[np.inner(r_vec,v_vec,axis=1)<0] = 2*np.pi-nu[np.inner(r_vec,v_vec,axis=1)<0]

	# # Degenerate cases (equatorial and circular orbits)
	# raan[n_mag == 0] = 0.0
	# arg_peri[(e_mag == 0) | (n_mag == 0)] = 0.0
	# nu[e_mag == 0] = 0.0

	# return np.stack([a, e_mag, i, raan, arg_peri, nu], axis=1)

	element_indices = np.r_[0:6, 8:11]
	if states.ndim == 1:
		elements = spice.oscltx(np.ascontiguousarray(states), time, mu)[element_indices]
	else:
		elements = np.zeros([len(states), 9])
		for i, (ti, si) in enumerate(zip(time, states)):
			element = spice.oscltx(np.ascontiguousarray(si), ti, mu)[element_indices]
			elements[i, :] = element

	return elements[:,(7,1,2,3,4,6)]