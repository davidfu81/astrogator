import numpy as np
from typing import Protocol

from eoms import EOM

class Integrator(Protocol):
	def integrate(self, y0:np.ndarray, t0:float, tf:float, step:float)->tuple[np.ndarray, np.ndarray]:
		pass