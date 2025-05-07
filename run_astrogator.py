import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from astrogator import *
from eoms import *
from integrators import *
from bodies import *
from helpers import visualize_trajectories

import time
if __name__ == "__main__":
	# load in leap seconds kernel
	spice.furnsh("./data/naif0012.tls")

	earth = Body(mu=0.39860e6, radius=6378.1)
	earth_two_body = Two_Body_EOM(earth.mu)
	initial_state = np.array([
		20000, # a (km)
		0.7, # e
		np.radians(45), # i (deg)
		np.radians(30), # raan (deg)
		np.radians(30), # arg peri(deg)
		np.radians(0), # true anom (deg)
	])
	prop_time = 3600*24*16

	# RKDP54
	start_time = time.time()
	astrogator_RKDP54 = Astrogator(initial_state, "2025 MAY 01 00:00:00 UTC", earth)
	astrogator_RKDP54.add_segment(
		Propagate(
			prop_time,
			integrator=RKDP54(eom=earth_two_body, abs_tol=1e-3,rel_tol=1e-4)
		)
	)
	astrogator_RKDP54.run_all()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Time taken to execute RKDP54 simulation: {elapsed_time} seconds")

	# RK4
	start_time = time.time()
	astrogator_RK4 = Astrogator(initial_state, "2025 MAY 01 00:00:00 UTC", earth)
	astrogator_RK4.add_segment(
		Propagate(prop_time, integrator=RK4(eom=earth_two_body))
	)
	astrogator_RK4.run_all()
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Time taken to execute RK4 simulation: {elapsed_time} seconds")


	# Analytical
	astrogator_analytical = Astrogator(initial_state, "2025 MAY 01 00:00:00 UTC", earth)
	astrogator_analytical.add_segment(
		Propagate(prop_time, integrator=Two_Body_Analytical(earth))
	)
	astrogator_analytical.run_all()

	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Time taken to execute simulation: {elapsed_time} seconds")

	visualize_trajectories(
		filename="test",
		save_to_dir="output",
		time=astrogator_analytical.time,
		state_arrays=[astrogator_RKDP54.states, astrogator_RK4.states, astrogator_analytical.states],
		mycolors=['LimeGreen', 'IndianRed', 'Gold'],
		trace_names=['RKDP54', 'RK4','Analytical'],
		cen_body_radius=earth.radius
	)

	# Compare Integration methods
	assert np.all(astrogator_analytical.time == astrogator_RKDP54.time)
	RKDP54_error = np.linalg.norm(astrogator_RKDP54.states[:,:3]-astrogator_analytical.states[:,:3],axis=1)
	RK4_error = np.linalg.norm(astrogator_RK4.states[:,:3]-astrogator_analytical.states[:,:3],axis=1)
	mission_time = (astrogator_analytical.time-astrogator_analytical.time[0])/3600
	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'Mission Time [hr]', fontsize=16)
	ax.set_ylabel(r'Position Error [km]', fontsize=16)
	ax.plot(mission_time, RKDP54_error, color='LimeGreen', alpha=1, linewidth=1, linestyle='-', label='RKDP54 Error')
	ax.plot(mission_time, RK4_error, color='IndianRed', alpha=1, linewidth=1, linestyle='-', label='RK4 Error')
	ax.legend(loc="upper left")
	fig.savefig("output/errors.png")
	
	plt.close()
