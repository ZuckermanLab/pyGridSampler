import grid_tools as gt 
import grid_sampler as gs
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt




def plot_grid(true_grid, pred_grid, x1_marginal, x2_marginal, x1_true, x2_true, title, fname):
    # Create a 4x4 subplot
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    # Top right: image heatmap of grid values
    axs[0, 1].imshow(true_grid,  cmap='hot', interpolation='nearest', extent=[x1_marginal.min(), x1_marginal.max(), x2_marginal.min(), x2_marginal.max()])
    axs[0, 1].axvline(x=x2_true, color='black', linestyle='--')
    axs[0, 1].axhline(y=x1_true, color='black', linestyle='--')


    # Top left: scatter plot of grid values
    axs[0, 0].scatter(pred_grid[:, 1], pred_grid[:, 0])
    axs[0, 0].axvline(x=x2_true, color='black', linestyle='--')
    axs[0, 0].axhline(y=x1_true, color='black', linestyle='--')

    # Bottom right: histogram of parameter 2
    axs[1, 0].hist(x2_marginal, density=True, range=x_bounds[1], bins=20)
    axs[1, 0].axvline(x=x2_true, color='black', linestyle='--')

    # Bottom left: histogram of parameter 1
    axs[1, 1].hist(x1_marginal, density=True, range=x_bounds[0], bins=20)
    axs[1, 1].axvline(x=x1_true, color='black', linestyle='--')



    # Add titles and axis labels
    axs[0, 1].set_title('Heatmap of log-likelihood')
    axs[0, 0].set_title('Scatter plot of grid values')
    axs[1, 1].set_title('Histogram of parameter 1')
    axs[1, 0].set_title('Histogram of parameter 2')

    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')

    axs[1, 0].set_xlabel('Parameter 1')
    axs[1, 0].set_ylabel('Density')

    axs[1, 1].set_xlabel('Parameter 2')
    axs[1, 1].set_ylabel('Density')

    plt.suptitle(f'{title}')
    plt.tight_layout()
    plt.savefig(f'{fname}')



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
    delta = 0.01 # keep smallest set of grid points whose weights sum to 1-delta. larger delta --> remove more gridpoints

    # create sampler class
    sampler = gs.GridSampler(calc_log_likelihood,args_list,y_obs, x_bounds, x_shifts)

    # initialize and run sampler
    grid_resolution, data_size, grid, spacing, log_likelihoods, rel_prob, weights, ess  = sampler.initialize_and_sample(init_grid_resolution, init_data_size, ess_min, delta, n_processes=4, max_iter=100)

    true_grid,  _, _, _, _, _  = sampler.calc_naive_grid(300, data_size-1)

    plot_grid(true_grid, grid, x1_marginal=np.array(gt.get_marginals_from_grid(grid)[0]), x2_marginal=np.array(gt.get_marginals_from_grid(grid)[1]), x1_true=x_truths[0], x2_true=x_truths[1], title='Grid results', fname='grid_results.png')