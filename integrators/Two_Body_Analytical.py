import numpy as np

from bodies import Body

class Two_Body_Analytical:
	"""
	This class implements propagation of a spacecraft state
	using the analytical solution of two body equations of motion
	"""
	def __init__(self, cen_body:Body):
		self.cen_body = cen_body

	def integrate(self, y0:np.ndarray, t0:float, tf:float, step:float)->tuple[np.ndarray, np.ndarray]:
		initial_oscelt = self.cen_body.convert_states_to_oscelt(t0, y0)
		orbital_period = self.cen_body.convert_sma_to_period(initial_oscelt[0]/(1-initial_oscelt[1]))

		time = np.arange(t0, tf, step)
		if tf not in time:
			time = np.concatenate((time, [tf]))
		states = np.zeros((len(time),6))
		states[0] = y0
		
		for i, ti in enumerate(time):
			oscelt_i = initial_oscelt.copy()
			oscelt_i[5] += (ti-t0)/orbital_period*2*np.pi
			oscelt_i[5] = np.mod(oscelt_i[5], 2*np.pi)
			oscelt_i[6] = ti
			state_i = self.cen_body.convert_oscelt_to_states(ti, oscelt_i)
			states[i] = state_i
		
		return states,time