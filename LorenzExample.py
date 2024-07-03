#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:24:55 2024

@author: mm17ktn
"""

from math import log2, ceil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import periodogram
import Functions as Fn

#%% Generate Lorenz
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 10#16
rho = 28# 45.92
beta = 8/3 #4

# Initial conditions
initial_state = [1.0, 1.0, 1.0]

# Time span
t_start = 0.0
t_end = 100
num_points = 10000
t_span = np.linspace(t_start, t_end, num_points)
print((t_end-t_start)/num_points)

# Solve the differential equations
solution = solve_ivp(lorenz_system, [t_start, t_end], initial_state,
                     args=(sigma, rho, beta), t_eval=t_span)

# Extract the solution
x_values = solution.y[0]
y_values = solution.y[1]
z_values = solution.y[2]

# Plot the attractor (Lorenz butterfly)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()

fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(t_span[5000:], x_values[5000:])
ax.set_xlabel('time $(i \Delta t)$', fontsize=18)
ax.set_ylabel('x', fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
plt.show()


x_obs = x_values[5000:] #ke[:10000]
dt = 0.01

#%% Cao's Algorithm for Embedding Dimension
Sturges = ceil(log2(len(x_obs)) +1)
Sturges

taus = range(1,501)
num_bins = Sturges
mi_values = [Fn.mutual_information(x_obs, num_bins, tau) for tau in taus]

# Plot mutual information
plt.plot(taus, mi_values, marker='o')
plt.xlabel('τ')
plt.ylabel('I(τ)')
plt.title('Mutual Information I(τ) vs. Time Delay τ')
plt.show()

# Find the first local minimum
for i in range(1, len(mi_values) - 1):
    if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
        first_local_min_tau = taus[i]
        first_local_min_value = mi_values[i]
        break

print(f"The first local minimum occurs at τ = {first_local_min_tau} with I(τ) = {first_local_min_value}")

tau = first_local_min_tau

# Using Cao's method to find the optimal embedding dimension
time_delay = tau
max_m = 20
E, E_star = Fn.cao_method(x_obs, 21, tau)

E1 = Fn.E1_ratio(E)
E2 = Fn.E2_ratio(E_star)

m_values = np.arange(1,21)
fig, ax = plt.subplots(1, figsize=(8,8), tight_layout=True)
ax.plot(m_values, E1, label='E1', marker='o')
ax.plot(m_values, E2, label='E2', marker='o')
ax.grid()
ax.legend()
ax.set_xlabel('m')
ax.set_ylabel('E1 and E2')
plt.show()

#%% Rosenstein's Algorithm for LLE
J_value = Fn.J_from_autocorrelation(x_obs)

# LLE
J = J_value
m = 3
t_end = 200
time_steps = 300 
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
ax.set_xlim(-10,200)
plt.show()

slope_start = int(0)
slope_end =int(2.5//dt)
slope, intercept, r_value, p_value, std_err = linregress(time_values[slope_start:slope_end], mean_log_distance[slope_start:slope_end])
print(slope)

end_val = int(3//dt)
best_fit = slope*time_values[:end_val] + intercept
fig, ax = plt.subplots(1,figsize=(8,6), tight_layout=True)
ax.plot(time_values[:], mean_log_distance[:], 'b-', label='mean log distance')
ax.plot(time_values[:end_val], best_fit, linestyle='--', color='orange', label='line of best fit')
ax.set_xlabel('Time $(i\Delta t)$', fontsize=18)
ax.set_ylabel('ln $\hat{d}$', fontsize=18)
#plt.title('Mean Log Distance over Time')
ax.grid()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
#plt.title('Mean Log Distance over Time')
ax.legend(fontsize=16)
ax.set_xlim(-10,200)
#ax.set_ylim(-7,-5)
plt.show()

print('LLE =', slope)
print('Lyapunov Time =', 1/slope)
