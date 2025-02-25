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
sys.stdout.reconfigure(line_buffering=True)

from docopt import docopt
args = docopt(__doc__)

input_path = args['--input_path']
output_path = args['--output_path']

#### load in larger data 5000-30000 hf ####
total_num_snapshots = 2500 #snap
x = np.load(input_path+'/x.npy')
z = np.load(input_path+'/z.npy')

with h5py.File(input_path+'/data_4var_5000_30000.h5', 'r') as df:
    time_vals = np.array(df['total_time_all'][:total_num_snapshots])
    q = np.array(df['q_all'][:total_num_snapshots])
    q = np.squeeze(q, axis=2)
    print('shape of data:', np.shape(q))

dt = time_vals[1] - time_vals[0]
print('dt:', dt)

x_sample = np.linspace(0,256,16)
x_sample = x_sample.astype(int)
print('sample space', len(x_sample))
z_pos = 32

max_m = 16
derv_threshold = 0.005

### PART 1 ###
E1_vals        = np.zeros((len(x_sample), max_m))
E2_vals        = np.zeros((len(x_sample), max_m))
tau_vals       = np.zeros(len(x_sample))
optimal_m_vals = np.zeros(len(x_sample))

for index, x_pos in enumerate(x_sample):
    print('index:', index)
    print('x_pos', x_pos)
    x_obs = q[:, x_pos, z_pos]

    #%% Cao's Algorithm for Embedding Dimension
    Sturges = ceil(log2(len(x_obs)) +1)
    Sturges
    
    taus = range(1,501)
    num_bins = Sturges
    mi_values = [Fn.mutual_information(x_obs, num_bins, tau) for tau in taus]
    
    # Plot mutual information
    fig, ax = plt.subplots(1, figsize=(8,6), constrained_layout=True)
    ax.plot(taus, mi_values, marker='o')
    ax.set_xlabel('tau')
    ax.set_ylabel('I(tau)')
    ax.grid()
    fig.savefig(output_path+'tau%i.png' % x_pos)
    
    # Find the first local minimum
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            first_local_min_tau = taus[i]
            first_local_min_value = mi_values[i]
            break
    
    print("The first local minimum occurs at tau = {} with I(tau) = {}".format(first_local_min_tau, first_local_min_value))
    
    tau = first_local_min_tau
    print('found tau for index:', index, 'tau=', tau)
    tau_vals[index] = tau
    
    # Using Cao's method to find the optimal embedding dimension
    time_delay = tau
    E, E_star = Fn.cao_method(x_obs, max_m+1, tau)
    
    E1 = Fn.E1_ratio(E)
    E2 = Fn.E2_ratio(E_star)
    
    m_values = np.arange(1,max_m+1)
    fig, ax = plt.subplots(1, figsize=(8,8), tight_layout=True)
    ax.plot(m_values, E1, label='E1', marker='o')
    ax.plot(m_values, E2, label='E2', marker='o')
    ax.grid()
    ax.legend()
    ax.set_xlabel('m')
    ax.set_ylabel('E1 and E2')
    fig.savefig(output_path+'/E1E2%i.png' % x_pos)
    
    E1_vals[index, :] = E1
    E2_vals[index, :] = E2
    
    np.save(output_path+'/E1.npy', E1_vals)
    np.save(output_path+'/E2.npy', E2_vals)
    np.save(output_path+'/tau_vals.npy', tau_vals)

    dE1 = np.diff(E1)
    stable_m = np.where(np.abs(dE1) < derv_threshold)[0]
    optimal_m = stable_m[0] + 2
    optimal_m_vals[index] = optimal_m
    np.save(output_path+'/optimal_m_vals.npy', optimal_m_vals)
    
    fig, ax = plt.subplots(1, figsize=(8,8), tight_layout=True)
    ax.plot(m_values[1:], dE1, label='dE1', marker='o')
    ax.axvline(optimal_m, color='r', linestyle='--', label=f'Optimal m = {optimal_m}')
    ax.grid()
    ax.legend()
    ax.set_xlabel('m')
    ax.set_ylabel('dE1')
    fig.savefig(output_path+'/dE1%i.png' % x_pos)
'''
'''

### PART 2 after calculating m values###
m_values = np.load(output_path+'/optimal_m_vals.npy')
m_values = m_values.astype(int) 
t_end = 700
curves = np.zeros((len(x_sample), t_end+1))
for index, x_pos in enumerate(x_sample):
    x_obs = q[:, x_pos, z_pos]

    #%% Rosenstein's Algorithm for LLE
    J_value = Fn.J_from_autocorrelation(x_obs)
    
    # LLE
    J          = J_value
    m          = m_values[index]
    time_steps = t_end + 100 
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
    fig.savefig(output_path+'/LLE_plot_no_lobf_%i.png' % x_pos)
    curves[index, :] = mean_log_distance
    
np.save(output_path+'/time_innovation.npy', time_innovation)
np.save(output_path+'/LLE_curves.npy', curves)


#### PART 3
time_innovation = np.load(output_path+'/time_innovation.npy')
curves          = np.load(output_path+'/LLE_curves.npy')
time_values     = time_innovation*dt
stable_times    = np.zeros(len(x_sample))
LLEs            = np.zeros(len(x_sample))
LT              = np.zeros(len(x_sample))

for i in range(len(x_sample)):
    curve = curves[i, :]
    dcurve = np.diff(curve)
    np.shape(dcurve)
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
            
    stable_time = time_values[stable_point] + 1
    print(stable_time)
    stable_times[i] = stable_time
    fig, ax = plt.subplots(1, figsize=(12,6), tight_layout=True)
    ax.plot(time_values[1:], dcurve, label='dcurve', marker='o')
    ax.axvline(stable_time, color='r', linestyle='--')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time  $(i\Delta t)$')
    ax.set_ylabel('d(ln $\hat{d}$)')
    fig.savefig(output_path+'/d_curve%i.png' % x_sample[i])
    
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
    fig.savefig(output_path+'/LLE_mean_plot_lobf_%i.png' % x_sample[i])
    
    print('LLE =', slope)
    print('Lyapunov Time =', 1/slope)
    LLEs[i] = slope
    LT[i] = 1/slope
    
print(LLEs)
print(LT)



#### PART 3 after calculating the curves ####
time_innovation = np.load(output_path+'/time_innovation.npy')
curves          = np.load(output_path+'/LLE_curves.npy')

mean_curves = np.mean(curves, axis = 0)
time_values = time_innovation*dt

fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], mean_curves[:], 'b-')
ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
ax.set_ylabel('ln $\hat{d}$', fontsize=18)
#plt.title('Mean Log Distance over Time')
ax.grid()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
fig.savefig(output_path+'/LLE_mean_plot_no_lobf.png')

dcurve = np.diff(mean_curves)
np.shape(dcurve)
derv_threshold_curve = 0.001
stable_point = np.where(np.abs(dcurve) < derv_threshold_curve)[0][0]
print(stable_point)
stable_time = time_values[stable_point] + 1
print(stable_time)
fig, ax = plt.subplots(1, figsize=(12,6), tight_layout=True)
ax.plot(time_values[1:], dcurve, label='dcurve', marker='o')
ax.axvline(stable_time, color='r', linestyle='--')
ax.grid()
ax.legend()
ax.set_xlabel('Time  $(i\Delta t)$')
ax.set_ylabel('d(ln $\hat{d}$)')
fig.savefig(output_path+'/dmean_curve.png')


slope_start = int(0)
slope_end =int(400//dt)
slope, intercept, r_value, p_value, std_err = linregress(time_values[slope_start:slope_end], mean_curves[slope_start:slope_end])
print(slope)

end_val = int(900//dt)
best_fit = slope*time_values[:end_val] + intercept
fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], mean_curves[:], 'b-', label='mean log distance')
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

print('LLE_mean_curve =', slope)
print('Lyapunov Time mean curve =', 1/slope)

print(LLEs)
print(LT)

print('mean LLEs', np.mean(LLEs))
print('mean LT', np.mean(LT))