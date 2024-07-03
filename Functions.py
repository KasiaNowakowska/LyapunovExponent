#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:46:35 2024

@author: mm17ktn
"""
from math import log
import numpy as np
import h5py
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
from scipy.signal import periodogram

# %% AMI

def calculate_probabilities(data, bins):
    """
    Calculate the probability distribution of the data for given bins.
    
    Parameters:
    data (array-like): The data to be binned.
    bins (int): Number of bins.
    
    Returns:
    tuple: Tuple containing:
        - prob (dict): Probability distribution p_i.
        - bin_edges (ndarray): Bin edges used for discretizing the data.
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    total_samples = len(data)
    prob = {i: count / total_samples for i, count in enumerate(hist)}
    
    return prob, bin_edges

def calculate_joint_probabilities(data, bins, tau):
    """
    Calculate the joint probability distribution p_ij(τ).
    
    Parameters:
    data (array-like): The data to be binned.
    bins (int): Number of bins.
    tau (int): Time delay.
    
    Returns:
    dict: Joint probability distribution p_ij(τ).
    """
    x_t = data[:-tau]
    x_t_tau = data[tau:]
    
    hist2d, xedges, yedges = np.histogram2d(x_t, x_t_tau, bins=bins, density=False)
    total_samples = len(x_t)
    joint_prob = {(i, j): count / total_samples for i, row in enumerate(hist2d) for j, count in enumerate(row)}
    
    return joint_prob

def mutual_information(data, bins, tau):
    """
    Calculate the actual mutual information I(τ) for time delay τ.
    
    Parameters:
    data (array-like): The data for which to calculate mutual information.
    bins (int): Number of bins for discretization.
    tau (int): Time delay.
    
    Returns:
    float: Actual mutual information I(τ).
    """
    # Define sets A and B
    A = data[:-tau]
    B = data[tau:]

    # Calculate marginal probabilities p_i
    prob_A, bin_edges_A = calculate_probabilities(A, bins)
    prob_B, bin_edges_B = calculate_probabilities(B, bins)
    
    # Calculate joint probabilities p_ij(τ)
    joint_prob_ij = calculate_joint_probabilities(data, bins, tau)

    # Calculate mutual information I(τ)
    mi = 0.0

    for (i, j), p_ij in joint_prob_ij.items():
        if p_ij > 0:  # To avoid log(0)
            p_i = prob_A[i]
            p_j = prob_B[j]
            mi += p_ij * log(p_ij / (p_i * p_j))

    return mi

#%% Caos Method

def distance_cao(xe, xi):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    xe (array): The first vector representing a position in reconstructed phase space.
    xi (array): The second vector representing a position in reconstructed phase space.
    
    Returns:
    float: Euclidean distance bewteen xi and xe.
    """
    return np.sqrt(np.sum((xi - xe)**2))

def get_nearest_neighbour_cao(xi, X, X_mplus1):
    """
    Calculates the nearest neighbour of xi.
    
    Parameters:
    xi (array-like): The vector representing a position in reconstructed phase space.
    X (matrix): The matrix of reconstructed phase space vectors with embedding dimenion m
    X_mplus1 (matrix): The matrix of reconstructed phase space vectors with embedding dimension m+1
    
    Returns:
    float: the index nn(i,m) representing the nearest neighbour to xi 
    """
    distances = [distance_cao(X[:, xe], X[:, xi]) for xe in range(X_mplus1.shape[1])]
    #print(distances)
    distances_removed_index = np.delete(distances, xi)
    #distances_removed_index = np.delete(distances_removed_index, -1)
    min_index = np.argmin(distances_removed_index)
    if min_index >= xi:
        min_index = min_index+1
    else:
        min_index = min_index
    return min_index

def cao_method(ts, max_dimension, time_delay):
    """
    Calculates E and E* from Cao's Algorithm
    
    Parameters:
    ts (array): Time series.
    max_dimension (int): The maximum embedding dimenion to invetsigate.
    time_delay (float): The time delay tau
    
    Returns:
    E (array): the value of E(m) for m=1,..,max_dimension
    E* (array): the value of E*(m) for m=1,...,max_dimension 
    """
    N = len(ts)
    E = np.zeros(max_dimension)
    E_star = np.zeros(max_dimension)
    
    for m in range(1, max_dimension + 1):
        print('m =', m)
        # Create delay embedded vectors
        X_m = np.array([ts[i:i+(m-1)*time_delay+1:time_delay] for i in range(N-(m-1)*time_delay)]).T
        X_mplus1 = np.array([ts[i:i+(m)*time_delay+1:time_delay] for i in range(N-(m)*time_delay)]).T
        print('shape of X_m', np.shape(X_m))
        print('shape of X_m+1', np.shape(X_mplus1))
        print(X_m.shape[1]-1)
        # Find nearest neighbors
        nearest_neighbors = [get_nearest_neighbour_cao(i, X_m, X_mplus1) for i in range(X_mplus1.shape[1])]
        
        # Calculate mean distance ratios E(m)
        sum_ratios = 0
        sum_ratios2 = 0
        for t in range(X_mplus1.shape[1]-1):
            t_prime = nearest_neighbors[t]
            print('t=', t, 't_prime=', t_prime)
            a = distance_cao(X_mplus1[:,t], X_mplus1[:,t_prime])/distance_cao(X_m[:,t], X_m[:,t_prime])
            sum_ratios += a
            a_star = distance_cao(ts[t+m*time_delay], ts[t_prime+m*time_delay] )
            sum_ratios2 += a_star
        
        E[(m - 1)] = sum_ratios / X_mplus1.shape[1]
        E_star[(m-1)] = sum_ratios2 / X_mplus1.shape[1]
    
    return E, E_star

def E1_ratio(E):
    """
    Calculates E1
    
    Parameters:
    E (array): array of values of E(m) for m=1,..,max_dimension
    
    Returns:
    float: the value of E1(m) for m=1,..,max_dimension
    """
    return E[1:]/E[:-1]

def E2_ratio(E_star):
    """
    Calculates E2
    
    Parameters:
    E_star (array): array of values of E*(m) for m=1,..,max_dimension
    
    Returns:
    float: the value of E2(m) for m=1,..,max_dimension
    """
    return E_star[1:]/E_star[:-1]

#%% LLE code

def distance(xe, xi):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    xe (array): The first vector representing a position in reconstructed phase space.
    xi (array): The second vector representing a position in reconstructed phase space.
    
    Returns:
    float: Euclidean distance bewteen xi and xe.
    """
    return np.sqrt(np.sum((xi - xe)**2))

def get_nearest_neighbour(xi, X, mu, time_steps):
    """
    Calculates the nearest neighbour of xi.
    
    Parameters:
    xi (array-like): The vector representing a position in reconstructed phase space.
    X (matrix): The matrix of reconstructed phase space vectors with embedding dimension m
    mu (float): The time period for which to not take a nearest neighbour (so they are not too close in time)
    time_steps (int): Specifies the range in time to look for a nearest neighbour (has to be large enough to find a 
                                                                                   nearest neighbour thats close, but 
                                                                                   means we don't need 
                                                                                   to search through all the points)
    
    Returns:
    float: the index representing the nearest neighbour to xi 
    """
    #print(X[xi])
    xes = np.arange(len(X) - time_steps)  # Indices for potential nearest neighbors within a specified range
    #print('xes', xes)
    # Calculate distances to potential neighbors
    ds = np.array([distance(X[xe], X[xi]) for xe in xes])
    #print(xi, 'ds=', ds)
    #print(len(ds))
    # Set distance to infinity if it's the same vector or too close based on muu
    ds = np.where(ds == 0, np.inf, ds)
    ds = np.where(np.abs(xi - xes) < mu, np.inf, ds)
    #print(xi, np.argmin(ds))
    return np.argmin(ds)

def get_nearest_neighbours(X, mu, time_steps):
    """
    Calculates the nearest neighbour of for all xi in X.
    
    Parameters:
    X (matrix): The matrix of reconstructed phase space vectors with embedding dimension m
    mu (float): The time period for which to not take a nearest neighbour (so they are not too close in time)
    time_steps (int): Specifies the range in time to look for a nearest neighbour (has to be large enough to find a 
                                                                                   nearest neighbour thats close, but 
                                                                                   means we don't need 
                                                                                   to search through all the points)
    
    Returns:
    array: an array of all the indicies representing the nearest neighbour to xi 
    """
    gnn = [get_nearest_neighbour(i, X, mu, time_steps) for i in range(len(X))]
    return gnn

def mean_period(ts):
    """
    Calculates the mean period of the timeseries.
    
    Parameters:
    ts (array): Time series.
    
    Returns:
    float: the mean period for the time series.
    """
    freq, spec = periodogram(ts)
    w = spec / np.sum(spec)
    mean_frequency = np.sum(freq * w)
    return 1 / mean_frequency

def lyap(ts, J, m, t_end, time_steps):
    """
    Calculates the (average) largest lyapunov exponent using Rosenstein's Algorithm.
    
    Parameters:
    ts (array): Time series.
    J (int): Lag.
    m (int): embedding dimension.
    t_end (int): The number of time steps to project into the future.
    time_steps (int): Specifies the range in time to look for a nearest neighbour (has to be large enough to find a 
                                                                                   nearest neighbour thats close, but 
                                                                                   means we don't need 
                                                                                   to search through all the points)
    
    Returns:
    time_innovation (array): array of the timesteps into the future (len: t_end)
    mean_log_distance (array): the mean log distance across all xi (len: t_end)
    distance_log_i (array): the log distance for each xi 
    """
    if time_steps < t_end:
        print("x is not greater than y. Stopping function.")
        return
    N = len(ts)
    M = N - (m - 1) * J
    print('J:', J, 'm:', m, 'N:', N, 'M:', M)
      
    X = np.full((M,m), np.nan)
      
    # Populate matrix X based on the loop logic
    for i in range(M):
        idx = np.arange(i, i + (m - 1) * J + 1, J)
        X[i, :] = ts[idx]
      
    j = get_nearest_neighbours(X, mu=mean_period(ts), time_steps=time_steps)
      
    #### estimate mean rate of seperation of nearest neighbours ####
    def expected_log_distance(i, X):
        n = len(X)
        d_ji = np.array([distance(X[j[k] + i], X[k + i]) for k in range(1, n - i)])
        log_only = np.log(d_ji)
        mean = np.mean(np.log(d_ji))
        return mean, log_only
      
    mean_log_distance = [expected_log_distance(i, X)[0] for i in range(t_end + 1)]
    distance_log_i = [expected_log_distance(i, X)[1] for i in range(t_end + 1)]
    time_innovation = np.arange(t_end + 1)
      
    return time_innovation, mean_log_distance, distance_log_i

def J_from_autocorrelation(ts):
    """
    Calculates the lag J to use in Rosenstein's Algorithm.
    
    Parameters:
    ts (array): Time series.
    
    Returns:
    float: The value of J
    """
    # Compute the autocorrelation function
    autocorr_result = np.correlate(ts-np.mean(ts), ts-np.mean(ts), mode='full')
    autocorr = autocorr_result[len(autocorr_result) // 2:]
    autocorr /= autocorr[0] #normalise by autocorrelation at lag 0

    # Find the lag (embedding delay) where the autocorrelation drops below 1 - 1/e of its initial value
    threshold = 1 - 1/np.exp(1)  # 1 - 1/e is approximately 0.632

    # Find the first lag where autocorrelation drops below the threshold
    J_value = np.where(autocorr < threshold)[0][0]
    return J_value
