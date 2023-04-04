import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys 
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from pyGridSampler import grid_sampler as gs
from pyGridSampler import grid_tools as gt

def plot_2d_grid(grid, x1_bounds, x2_bounds,fname, x1_true=3, x2_true=5):
    # reduce grid  
    fig = plt.figure(figsize=(10,7))
    # scatter plot of grid points
    plt.title('active grid points')
    plt.xlabel('x2')
    plt.ylabel('x1')
    plt.xlim(x2_bounds)
    plt.ylim(x1_bounds)
    y = [x[0] for x in grid]
    x = [x[1] for x in grid]
    plt.plot(x,y, 'o', alpha=0.45)
    plt.axvline(x=x2_true, color='black', linestyle='--')
    plt.axhline(y=x1_true, color='black', linestyle='--')
    plt.savefig(f'{fname}')
    plt.close()


def calc_log_likelihood(X, args):
    # simple linear model: y = x_1*t + x_2
    # Normal log-likelihood with known mean and noise stdev.
    k = X[0]
    sigma = X[1]
    y_obs = args[0]
    t = args[1]
    y0 = 50
    y_pred = y0*np.exp(-k*t)
    log_likelihood = sp.stats.norm.logpdf(y_obs, y_pred, sigma).sum()
    #log_likelihood = np.sum(np.log(sp.stats.norm.pdf(y_pred,y_obs,sigma)))
    return log_likelihood


def create_tempering_list(log_like_args):
    # creates a list of arguments for the log-likelihood function at each tempering stage
    y_obs = log_like_args[0]
    t_full = log_like_args[1]
    n_data_points = len(y_obs)

    # get a list of data indices for data tempering - e.g. [[0], [0,1], [0,1,2]]
    # each list index corresponds to a data tempering stage
    idx_sublist = gt.get_sorted_idx_sublists(y_obs)

    # get y_obs and t lists for each data tempering stage
    # e.g. t_tempering_list = [t[0], t[0,1], t[0,1,2]]
    t_sublist = gt.get_sorted_sublists_from_idx(idx_sublist, t_full)
    y_obs_sublist = gt.get_sorted_sublists_from_idx(idx_sublist, y_obs)

    # at a given tempering stage n, this list contains data[:n] and t[:n] for the log-likelihood calculation
    args_tempering_list = [[y_obs_sublist[i], t_sublist[i]] for i in range(n_data_points)]
    return args_tempering_list


if __name__ == "__main__":
    # model configuration
    np.random.seed(0)
    k_true = 5
    sigma_true = 0.5
    t_full = np.linspace(0, 2, 10)

    y_true= 50*np.exp(-k_true*t_full)
    x_truths = [k_true, sigma_true]
    y_obs = y_true + np.random.normal(0,sigma_true,np.size(y_true))

    # plot data
    plt.figure(figsize=(10,7))
    plt.plot(y_true, label='y_true')
    plt.plot(y_obs, 'o', label='y_obs')
    plt.title('exponential model data: y=Ae^(-kt)')
    plt.ylabel('y')
    plt.xlabel('t')
    plt.legend()
    plt.savefig('exp_model_data.png')
    

    # configure sampler parameters
    x_bounds = [(1,7), (0.1,1)]
    x_shifts = [[1,1], [-1,1], [1,-1], [-1,-1], [1,0], [-1,0], [0,1], [0,-1]]
    args_list = create_tempering_list([y_obs, t_full])
    
    init_data_size = 2 # use 2 data points initially
    init_grid_resolution = 100 # 20 points per parameter -> total grid size 20^n_dim points
    ess_min = 1000  # target effective size threshold. larger ess_min --> denser grid 
    delta = 0.05 # keep smallest set of grid points whose weights sum to 1-delta. larger delta --> remove more gridpoints

    # create sampler class
    sampler = gs.GridSampler(calc_log_likelihood,args_list,y_obs, x_bounds, x_shifts)

    # initialize and run sampler
    grid_resolution, data_size, grid, spacing, log_likelihoods, rel_prob, weights, ess  = sampler.initialize_and_sample(init_grid_resolution, init_data_size, ess_min, delta, n_processes=4, max_iter=100)

    # plot 2d scatter of grid results
    fig = plt.figure(figsize=(10,7))
    plt.title('grid points')
    plt.xlabel('sigma')
    plt.ylabel('k')
    plt.xlim(x_bounds[1])
    plt.ylim(x_bounds[0])
    y = [x[0] for x in grid]
    x = [x[1] for x in grid]
    plt.plot(x,y, 'o', alpha=0.45)
    plt.axvline(x=sigma_true, color='black', linestyle='--')
    plt.axhline(y=k_true, color='black', linestyle='--')
    plt.savefig(f'grid_results_scatter.png')
    plt.close()

    # plot 1d marginals of grid results
    gt.plot_grid_marginals(grid, x_bounds, x_truths, 'grid_results_marginals.png')