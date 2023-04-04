#import adtgrid as grid
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import grid_tools as gt
import time


def linear_model(x1, x2, t):
    return np.array(x1*t + x2)

def update_log_likelihood_args(data_size, y_obs_sublist, t_sublist, sigma_true):
    return [y_obs_sublist[data_size], t_sublist[data_size], sigma_true]

def calc_log_likelihood(X, args):
    x1_pred = X[0]
    x2_pred = X[1]
    y_obs = args[0]
    t = args[1]
    sigma = args[2]
    y_pred = linear_model(x1_pred, x2_pred, t)
    log_likelihood = np.sum(np.log(sp.stats.norm.pdf(y_pred,y_obs,sigma)))
    return log_likelihood

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

def plot_2d_grid_scatter(results, results_grid, x1, x2, x1_bounds, x2_bounds, label, fname):



    min_val = np.min(results)
    max_val = np.max(results)
    results_scaled = (results - min_val)/(max_val-min_val)

    fig = plt.figure(figsize=(15, 5))

    # scatter plot of grid points
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.set_title('active grid points')
    ax0.set_xlabel('x2')
    ax0.set_ylabel('x1')
    xy_coords = np.array([(x2_val, x1_val) for x1_val in x1_points for x2_val in x2_points])
    ax0.scatter(xy_coords[:,0], xy_coords[:,1])
    ax0.set_aspect('equal')
   
    # heatmap of scaled log-likelihood
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(results_grid, cmap='hot', interpolation='nearest', extent=[x2.min(), x2.max(), x1.min(), x1.max()], vmin=np.min(results_grid), vmax=np.max(results_grid))
    ax1.set_title('scaled log-likelihood')
    ax1.set_xlabel('x2')
    ax1.set_ylabel('x1')
    ax1.axvline(x=x2_true, color='black', linestyle='--')
    ax1.axhline(y=x1_true, color='black', linestyle='--')

    # 3d meshgrid plot of scaled-loglikelihood
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    X1, X2 = np.meshgrid(x1, x2)
    ax2.plot_surface(X2, X1, results_grid, cmap='hot')
    ax2.set_title('scaled log-likelihood')
    ax2.set_xlabel('x2')
    ax2.set_ylabel('x1')
    ax2.set_zlabel('log-likelihood')
    plt.suptitle(f'{label}')
    plt.tight_layout()
    plt.savefig(f'{fname}')
    plt.close()


# def get_marginals_from_grid(grid):
#     grid_marginals = np.transpose(grid).tolist()
#     return grid_marginals


# def plot_grid_marginals(grid, x_bounds, truths, fname):
#     # create subplots
#     marginals = get_marginals_from_grid(grid)
#     fig, axs = plt.subplots(1, len(marginals), figsize=(8, 4))

#     # plot histograms
#     for i in range(len(marginals)):
#         if truths:
#             axs[i].hist(marginals[i], density=True, bins=50, range=x_bounds[i], histtype='step')
#             axs[i].axvline(truths[i], color='black', ls='--')
#         else:
#             axs[i].hist(marginals[i], density=True, bins=50, range=x_bounds[i], histtype='step' )
#         axs[i].set_xlabel(f'x_{i+1}')
#         axs[i].set_ylabel('density')

#     plt.title('grid 1D marginal distributions')
#     plt.tight_layout()
#     plt.savefig(fname)


# def eval_grid_points(grid, args, n_processes=4):
#     with Pool(processes=n_processes) as pool:
#         log_likelihoods = pool.starmap(calc_log_likelihood, [(X, args) for X in grid])
#     rel_prob = calc_rel_p(calc_rel_logp(log_likelihoods))
#     weights = calc_norm_weights(rel_prob)
#     ess = calc_ess(rel_prob)
#     return log_likelihoods, rel_prob, weights, ess


# def gen_grid_points(n_dim, x_bounds, grid_resolution): 

#     x_points = [np.linspace(bound[0], bound[1], grid_resolution) for bound in x_bounds] 
#     x_spacing = [np.abs(x[1]-x[0]) for x in x_points]
#     initial_grid = np.array(np.meshgrid(*x_points)).T.reshape(-1,n_dim)
#     return initial_grid, x_spacing


# def update_x_spacing(x_spacing, div_amount):
#     new_x_spacing = [x/div_amount for x in x_spacing]
#     return new_x_spacing


# def reduce_grid_points(grid, weights, delta):
#     weights_sorted_idx = np.argsort(weights)[::-1]
#     weights_sorted = weights[weights_sorted_idx]
#     cumulative_sum = np.cumsum(weights_sorted)
#     threshold_index = np.argwhere(cumulative_sum >= 1-delta)[0, 0]
#     indices_to_keep = weights_sorted_idx[:threshold_index+1]
#     reduced_grid = grid[indices_to_keep]  
#     return reduced_grid


# def check_grid_boundary(grid, x_bounds):
#     # ~O(n_dim*n_array_elements)

#     # remove points outside of boundary
#     n_dim = grid.shape[1]
#     boundary_mask = np.ones(grid.shape[0], dtype=bool)  # initial mask = True for all elements
#     for i in range(n_dim):
#         lb, ub = x_bounds[i]
#         boundary_mask &= ((grid[:, i] >= lb) & (grid[:, i] <= ub))  # update mask for each element
#     updated_grid = grid[boundary_mask]
#     return updated_grid


# def expand_and_pack_grid_points(grid, x1_bounds, x2_bounds, x1_spacing, x2_spacing):
#     # update by making more general for n_dim
#     expanded_grid = np.vstack((grid, 
#                                grid+np.array([x1_spacing, x2_spacing]), 
#                                grid+np.array([-x1_spacing, x2_spacing]),   
#                                grid+np.array([x1_spacing, -x2_spacing]),  
#                                grid+np.array([-x1_spacing, -x2_spacing])
#                                ))
#     print(np.shape(expanded_grid))

#     boundary_mask = ((expanded_grid[:, 0] >= x1_bounds[0]) & (expanded_grid[:, 0] <= x1_bounds[1])) & \
#        ((expanded_grid[:, 1] >= x2_bounds[0]) & (expanded_grid[:, 1] <= x2_bounds[1]))
#     expanded_grid = expanded_grid[boundary_mask]
#     return expanded_grid


# def add_grid_points(grid, x_bounds, x_shifts, x_spacing):


#     # expand grid points but only keep unique values 
#     expansion_list = [grid]
    
#     # expand grid points by adding neghiboring points based on a shift vector and shift amount
#     for x_shift in x_shifts:
#         expansion_i = grid + np.array(x_shift)*np.array(x_spacing)  # x_neighbor = x_direction*dx
#         expansion_i = np.unique(expansion_i, axis=0)  # remove duplicates ~O(nlogn)
#         expansion_list.append(expansion_i)
#     expanded_grid = np.vstack(expansion_list)
#     expanded_grid = check_grid_boundary(np.vstack(expansion_list), x_bounds)
#     return expanded_grid


# def calc_rel_logp(logp):
#     """ relative log probability (log-likelihood):
#     log-likelihood = log(p) --> log(p/pmax) = log(p) - ln(p_max)
#     """
#     return logp - np.max(logp)


# def calc_rel_p(rel_logp):
#     """ relative probability (likelihood):
#     rel_p = p/pmax = e**log(p/pmax)
#     """
#     return np.exp(rel_logp)


# def calc_norm_weights(p):
#     """ weight used for resampling (so the sum of pobability = 1):
#     sum(p) = 1 -> w = p/sum(p)
#     """
#     return p / np.sum(p)


# def calc_ess(rel_p):
#     """ effective sample size (ESS):
#     ESS = sum(p/pmax) = sum(relative p)
#     """
#     return np.sum(rel_p)


# def get_sorted_idx_sublists(data):
#     idx_list = list(range(len(data)))
#     idx_sublist = [idx_list[:i+1] for i in range(len(idx_list))] 
#     return idx_sublist
    

# def get_sorted_sublists_from_idx(idx_sublist, x):
#     x_sublist = [x[i] for i in idx_sublist]
#     return x_sublist


if __name__ == "__main__":

    # model configuration
    np.random.seed(0)
    x1_true = 3
    x2_true = 5
    sigma_true = 1
    x1_bounds = (1,5)  # (lower bound, upper bound)
    x2_bounds = (3,8)
    x_truths = [x1_true, x2_true]
    t_full = np.linspace(1,10,10)
    y_true = linear_model(x1_true, x2_true, t_full)
    y_obs = y_true + np.random.normal(0,sigma_true,np.size(y_true))

    x_bounds = [x1_bounds, x2_bounds]
    x_shifts = [[1,1], [-1,1], [1,-1], [-1,-1]]

    n_dim = len(x_bounds)
    n_data_points = len(y_obs)

    # get sublists to use for data tempering
    idx_sublist = gt.get_sorted_idx_sublists(y_obs)
    t_sublist = gt.get_sorted_sublists_from_idx(idx_sublist, t_full)
    y_obs_sublist = gt.get_sorted_sublists_from_idx(idx_sublist, y_obs)
    args_list = [[y_obs_sublist[i], t_sublist[i], sigma_true] for i in range(n_data_points)]

    
    print(args_list[:3])
    assert(1==0)

    init_data_size = 2 # use 2 data points initially
    init_grid_resolution = 10 # 10 points per parameter -> total grid size 50^n_dim

    ess_min = 100  # target effective size threshold for a 'valid sample'
    delta = 0.01 # 1-delta determines how many grid points to keep during reduction 

    # create initial grid and calculate initial probabilities and ess
    init_args = update_log_likelihood_args(init_data_size, y_obs_sublist, t_sublist, sigma_true)
    init_grid, init_spacing = gt.gen_grid_points(n_dim, x_bounds, init_grid_resolution)
    init_log_likelihoods, init_rel_prob, init_weights, init_ess = gt.eval_grid_points(init_grid, calc_log_likelihood, init_args)  

    # iteratively update initial grid until ess > ess_min
    ess = init_ess
    data_size = init_data_size
    grid_resolution = init_grid_resolution
    with tqdm(total=None, desc="Processing") as pbar:
        
        while ess < ess_min:
            grid_resolution = grid_resolution + 1  
            args = update_log_likelihood_args(data_size, y_obs_sublist, t_sublist, sigma_true)
            grid, x_spacing = gt.gen_grid_points(n_dim, x_bounds, grid_resolution)
            log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, calc_log_likelihood, args) 
            pbar.update()
            pbar.set_description(f"Intialization: data_size={data_size}, grid_resolution={grid_resolution}, n_grid_points={np.shape(grid)[0]}, ESS={ess}")
    active_grid = grid
    gt.plot_grid_marginals(active_grid, x_bounds, x_truths, 'test.png')

    # remove grid points with low likelihood values (cumulative sum of weights = 1-delta)
    reduced_grid = gt.reduce_grid_points(active_grid,weights,delta) 
    active_grid = reduced_grid

    # iterate through remaining data (data tempering)
    pbar = tqdm(range(init_data_size+1, n_data_points),desc='Processing:')
    for i in pbar:
        data_size = i
        args = update_log_likelihood_args(data_size, y_obs_sublist, t_sublist, sigma_true)
        expanded_grid = gt.add_grid_points(active_grid, x_bounds, x_shifts, x_spacing)
        active_grid = expanded_grid
        x_spacing = gt.update_x_spacing(x_spacing,2)
        packed_grid = gt.add_grid_points(active_grid, x_bounds, x_shifts, x_spacing)
        active_grid = packed_grid
        log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(active_grid,calc_log_likelihood, args) 
        

        while ess<ess_min:
            expanded_grid = gt.add_grid_points(active_grid, x_bounds, x_shifts, x_spacing)
            active_grid = expanded_grid
            x_spacing = gt.update_x_spacing(x_spacing,2)
            packed_grid = gt.add_grid_points(active_grid, x_bounds, x_shifts, x_spacing)
            active_grid = packed_grid
            log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(active_grid, calc_log_likelihood, args) 
            
        
        reduced_grid = gt.reduce_grid_points(active_grid, weights, delta)
        active_grid = reduced_grid
        pbar.set_description(f"Processing: data_size={data_size}, grid_resolution={grid_resolution}, n_grid_points={np.shape(grid)[0]}, ESS={ess}")

    gt.plot_grid_marginals(active_grid, x_bounds, x_truths, 'test.png')