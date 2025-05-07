import numpy as np
import spiceypy as spice

def kepler_to_cartesian(elts:np.ndarray, time:np.ndarray, mu:float):
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
	mu : float
		graviational parameter of central body (km**3/s**2)

	Output
	np.ndarray
		- (N,6) Cartesian states (pos, vel) in km, km/s
	"""
	# Obtain elements in form expected by spiceypy
	a = elts[:,0]
	e = elts[:,1]
	i = elts[:,2]
	raan = elts[:,3]
	argp = elts[:,4]
	true_anom = elts[:,5]

	rp = a*(1-e)
	
	# Mean anomaly
	r = a*(1-e**2)/(1+e*np.cos(true_anom))
	eccen_anom = np.arctan2(np.sqrt(1 - e**2) * np.sin(true_anom),
						e + np.cos(true_anom))
	mean_anom = eccen_anom - e*np.sin(eccen_anom)
	mean_anom = np.mod(mean_anom, 2*np.pi)

	# Build SPICE-compliant orbital elements array: (N,8)
	oscelt = np.stack([rp, e, i, raan, argp, mean_anom, time, mu * np.ones_like(a)], axis=1)

	# Convert to Cartesian
	if oscelt.ndim == 1:
		states = spice.conics(oscelt, time)
	else:
		states = np.zeros_like(oscelt)
		for i, (ti, ei) in enumerate(zip(time, oscelt)):
			state = spice.conics(ei, ti)
			states[i, :] = state

	return states