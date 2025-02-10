"""
python script for ESN grid search.

Usage: lyapunov.py [--input_path=<input_path> --output_path=<output_path>]

Options:
    --input_path=<input_path>          file path to use for data
    --output_path=<output_path>        file path to save images output [default: ./images]
"""

# import packages
from math import log2, ceil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import periodogram
import Functions as Fn
import h5py
import sys
import json
import os
sys.stdout.reconfigure(line_buffering=True)

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']

#### Load Data ####
q = np.load(input_path + '/q5000_30000.npy')
ke = np.load(input_path + '/KE5000_30000.npy')
total_time = np.load(input_path + '/total_time5000_30000.npy')

# Reshape the arrays into column vectors
ke_column = ke.reshape(len(ke), 1)
q_column = q.reshape(len(q), 1)
#evap_column = evap.reshape(len(evap), 1)

data = np.hstack((ke_column, q_column))

# Print the shape of the combined array
print(data.shape)

x_obs = q[:5000]
print(np.shape(x_obs))

fig, ax = plt.subplots(1, figsize=(12,3), constrained_layout=True)
ax.plot(x_obs)
ax.set_xlabel('data points')
ax.set_ylabel('data')
ax.grid()
fig.savefig(output_path+'/data.png')
plt.close()

max_m = 16
derv_threshold = 0.005

# Full path for saving the file
output_file = "parameters.txt"

output_path_par = os.path.join(output_path, output_file)

# New parameters to add
new_params = f"derv_threshold: {derv_threshold:.4f}\n"

# Append new parameters to the file
with open(output_path_par, "a") as file:
    file.write(new_params)


#%% Cao's Algorithm for Embedding Dimension
Sturges = ceil(log2(len(x_obs)) +1)
Sturges

taus = range(1,501)
num_bins = Sturges
mi_values = [Fn.mutual_information(x_obs, num_bins, tau) for tau in taus]
print('mi values', np.shape(mi_values))

# Plot mutual information
fig, ax = plt.subplots(1, figsize=(8,6), constrained_layout=True)
ax.plot(taus, mi_values, marker='o')
ax.set_xlabel('tau')
ax.set_ylabel('I(tau)')
ax.grid()
fig.savefig(output_path+'/tau.png')

# Find the first local minimum
for i in range(1, len(mi_values) - 1):
    if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
        first_local_min_tau = taus[i]
        first_local_min_value = mi_values[i]
        break

print("The first local minimum occurs at tau = {} with I(tau) = {}".format(first_local_min_tau, first_local_min_value))

tau = first_local_min_tau
print('found tau=', tau)

# New parameters to add
new_params = f"tau: {tau:.2f}\n"

# Append new parameters to the file
with open(output_path_par, "a") as file:
    file.write(new_params)

# Using Cao's method to find the optimal embedding dimension
time_delay = tau
E, E_star = Fn.cao_method(x_obs, max_m+1, tau)

E1 = Fn.E1_ratio(E)
E2 = Fn.E2_ratio(E_star)

m_values = np.arange(1,max_m+1)
fig, ax = plt.subplots(1, figsize=(8,6), tight_layout=True)
ax.plot(m_values, E1, label='E1', marker='o')
ax.plot(m_values, E2, label='E2', marker='o')
ax.grid()
ax.legend()
ax.set_xlabel('m')
ax.set_ylabel('E1 and E2')
fig.savefig(output_path+'/E1E2.png')

np.save(output_path+'/E1.npy', E1)
np.save(output_path+'/E2.npy', E2)
np.save(output_path+'/tau.npy', tau)

E1 = np.load(output_path+'/E1.npy')
E2 = np.load(output_path+'/E2.npy')
m_values = np.arange(1,max_m+1)

dE1 = np.diff(E1)
stable_m = np.where(np.abs(dE1) < derv_threshold)[0]
optimal_m = stable_m[0] + 1
np.save(output_path+'/optimal_m.npy', optimal_m)

fig, ax = plt.subplots(1, figsize=(8,8), tight_layout=True)
ax.plot(m_values[1:], dE1, label='dE1', marker='o')
ax.axvline(optimal_m, color='r', linestyle='--', label=f'Optimal m = {optimal_m}')
ax.grid()
ax.legend()
ax.set_xlabel('m')
ax.set_ylabel('dE1')
fig.savefig(output_path+'/dE1.png')

### PART 2 after calculating m values###
optimal_m = np.load(output_path+'/optimal_m.npy')
dt = total_time[1] - total_time[0]
print('m=', optimal_m)


t_end = 2000

#%% Rosenstein's Algorithm for LLE
J_value = Fn.J_from_autocorrelation(x_obs)

# LLE
J          = J_value
m          = optimal_m
time_steps = t_end + 100 

# New parameters to add
new_params = f"m: {optimal_m:.2f}\n J: {J:.2f}\n t_end: {t_end:.2f}\n time_steps: {time_steps:.2f}\n"

# Append new parameters to the file
with open(output_path_par, "a") as file:
    file.write(new_params)

time_innovation, mean_log_distance, distance_log_i = Fn.lyap(x_obs, J, m, t_end, time_steps)

time_values = time_innovation*dt
fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], mean_log_distance[:], 'b-')
ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
ax.set_ylabel('ln $\hat{d}$', fontsize=18)
#plt.title('Mean Log Distance over Time')
ax.grid()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
fig.savefig(output_path+'/LLE_plot_no_lobf.png')
    
np.save(output_path+'/time_innovation.npy', time_innovation)
np.save(output_path+'/LLE_curve.npy', mean_log_distance)


#### PART 3
optimal_m = np.load(output_path+'/optimal_m.npy')
dt = total_time[1] - total_time[0]
print('m=', optimal_m)
time_innovation = np.load(output_path+'/time_innovation.npy')
curve          = np.load(output_path+'/LLE_curve.npy')
time_values     = time_innovation*dt
t_end = 2000

'''
### removed this part and done by eye '####
dcurve = np.diff(curve)
print(np.shape(dcurve))
derv_threshold_curve = 0.02
stable_points = np.where(np.abs(dcurve) < derv_threshold_curve)[0]
print(stable_points)
Flag  = True
v = 0
while Flag == True:
    stable_index = stable_points[v]
    print(stable_index)
    if np.all(np.abs(dcurve[stable_index:stable_index+20]) < derv_threshold_curve):
        stable_point = stable_index
        Flag = False
    else:
        v += 1
        
stable_time = time_values[stable_point] 
print(stable_time)
stable_times[i] = stable_time
fig, ax = plt.subplots(1, figsize=(12,6), tight_layout=True)
ax.plot(time_values[1:], dcurve, label='dcurve', marker='o')
ax.axvline(stable_time, color='r', linestyle='--')
ax.grid()
ax.legend()
ax.set_xlabel('Time  $(i\Delta t)$')
ax.set_ylabel('d(ln $\hat{d}$)')
fig.savefig(output_path+'/d_curve.png')
'''

stable_time = 750
slope_start = int(0)
slope_end =int(stable_time//dt)
slope, intercept, r_value, p_value, std_err = linregress(time_values[slope_start:slope_end], curve[slope_start:slope_end])
print(slope)

end_val = int(stable_time//dt)
best_fit = slope*time_values[:end_val] + intercept
fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], curve[:], 'b-', label='mean log distance')
ax.plot(time_values[:end_val], best_fit, linestyle='--', color='orange', label='line of best fit')
ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
ax.set_ylabel('ln $\hat{d}$', fontsize=18)
#plt.title('Mean Log Distance over Time')
ax.grid()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
#plt.title('Mean Log Distance over Time')
ax.legend(fontsize=16)
fig.savefig(output_path+'/LLE_mean_plot_lobf.png')

print('LLE =', slope)
print('Lyapunov Time =', 1/slope)
np.save(output_path+'/LLE.npy', slope)
np.save(output_path+'/LT.npy', 1/slope)

# New parameters to add
new_params = f"stable_time: {stable_time:.2f}\n LLE: {slope:.6f}\n LT: {(1/slope):.2f}\n"

# Append new parameters to the file
with open(output_path_par, "a") as file:
    file.write(new_params)

