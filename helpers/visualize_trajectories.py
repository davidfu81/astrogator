import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os
import plotly.graph_objects as go

def visualize_trajectories(
		filename:str, save_to_dir:str, time, state_arrays, mycolors, trace_names, cen_body_radius:float
	):
	fig_r = plt.figure(figsize=(8, 6))
	ax_r = fig_r.add_subplot(111)
	ax_r.tick_params(axis='both', which='major', labelsize=12)
	ax_r.grid()
	ax_r.set_xlabel(r'T [hr]', fontsize=16)
	ax_r.set_ylabel(r'R [km]', fontsize=16)


	fig_xy = plt.figure(figsize=(8, 6))
	ax_xy = fig_xy.add_subplot(111)
	ax_xy.tick_params(axis='both', which='major', labelsize=12)
	ax_xy.grid()
	ax_xy.set_xlabel(r'X [km]', fontsize=16)
	ax_xy.set_ylabel(r'Y [km]', fontsize=16)

	fig_xz = plt.figure(figsize=(10, 6))
	ax_xz = fig_xz.add_subplot(111)
	ax_xz.tick_params(axis='both', which='major', labelsize=12)
	ax_xz.grid()
	ax_xz.set_xlabel(r'X [km]', fontsize=16)
	ax_xz.set_ylabel(r'Z [km]', fontsize=16)

	fig_yz = plt.figure(figsize=(10, 6))
	ax_yz = fig_yz.add_subplot(111)
	ax_yz.tick_params(axis='both', which='major', labelsize=12)
	ax_yz.grid()
	ax_yz.set_xlabel(r'Y [km]', fontsize=16)
	ax_yz.set_ylabel(r'Z [km]', fontsize=16)

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	ax.tick_params(axis='x', labelsize=12)
	ax.tick_params(axis='y', labelsize=12)
	ax.tick_params(axis='z', labelsize=12)
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, 'grid_steps': 1e3})
	ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, 'grid_steps': 1e3})
	ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, 'grid_steps': 1e3})

	# Spacecraft States
	for color, state_array, trace_name in zip(mycolors, state_arrays, trace_names):
		ax.plot(state_array[:,0], state_array[:,1], state_array[:,2], color=color, alpha=1, linewidth=1, linestyle='-', label=trace_name)
		ax_xy.plot(state_array[:,0], state_array[:,1], color=color, alpha=1, linewidth=1, linestyle='-', label=trace_name)
		ax_xz.plot(state_array[:,0], state_array[:,2], color=color, alpha=1, linewidth=1, linestyle='-', label=trace_name)
		ax_yz.plot(state_array[:,1], state_array[:,2], color=color, alpha=1, linewidth=1, linestyle='-', label=trace_name)
		ax_r.plot(time, np.linalg.norm(state_array[:,:3],axis=1), color=color, alpha=1, linewidth=1, linestyle='-', label=trace_name)

	# Central Body
	ax_xy.add_patch(patches.Circle((0,0), cen_body_radius, edgecolor='DarkBlue', facecolor='DarkBlue', linewidth=2))
	ax_xz.add_patch(patches.Circle((0,0), cen_body_radius, edgecolor='DarkBlue', facecolor='DarkBlue', linewidth=2))
	ax_yz.add_patch(patches.Circle((0,0), cen_body_radius, edgecolor='DarkBlue', facecolor='DarkBlue', linewidth=2))
	ax_xy.set_aspect('equal')
	ax_xz.set_aspect('equal')
	ax_yz.set_aspect('equal')
	ax.scatter([0], [0], [0], c='DarkBlue', s=50, alpha=0.7)

	# Save Figures
	ax_xy.legend()
	fig_xy.savefig(save_to_dir + "/" + filename + "_xy.png", dpi=300)

	ax_yz.legend()
	fig_yz.savefig(save_to_dir + "/" + filename + "_yz.png", dpi=300)

	ax_xz.legend(loc='lower left')
	fig_xz.savefig(save_to_dir + "/" + filename + "_xz.png", dpi=300)

	ax_r.legend()
	fig_r.savefig(save_to_dir + "/" + filename + "_r.png", dpi=300)

	fig.savefig(save_to_dir + "/" + filename + ".png", dpi=300)

	'''
	3D Figure
	'''
	fig = go.Figure()

	# Plot spacecraft ephemerides
	for color, state_array, trace_name in zip(mycolors, state_arrays, trace_names):
		fig.add_trace(go.Scatter3d(x=state_array[:,0], y=state_array[:,1], z=state_array[:,2],
							  mode='lines', line=dict(color=mcolors.to_hex(color), width=1), name=trace_name))
	

	# Plot Central Body
	num_points = 50  # Resolution of the sphere
	r = cen_body_radius  # Radius of the sphere
	theta = np.linspace(0, np.pi, num_points)  # Polar angle
	phi = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle
	theta, phi = np.meshgrid(theta, phi)
	x = r * np.sin(theta) * np.cos(phi)
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)

	# Create a 3D surface plot for the sphere
	fig.add_trace(go.Surface(
		x=x, y=y, z=z, colorscale=[(0, mcolors.to_hex('DarkBlue')), (1, mcolors.to_hex('DarkBlue'))], opacity=0.9, showscale=False))

	fig.update_layout(
	scene=dict(
		xaxis_title='X [km]',
		yaxis_title='Y [km]',
		zaxis_title='Z [km]',
		xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
		yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey'),
		zaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgrey')
	),
	margin=dict(l=0, r=0, b=0, t=0)
	)

	# Save to HTML
	fig.write_html(os.path.join(save_to_dir, filename + ".html"))