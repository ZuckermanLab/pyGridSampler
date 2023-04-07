import numpy as np
from tqdm import tqdm
import sys 
import os
import pyGridSampler.grid_tools as gt


class GridSampler:
    """Class for sampling grids using an adaptive data-tempering algorithm.

    Args:
        func (function): The log-likelihood function to evaluate the grid.
        args_list (list): A list of arguments to be passed to the log-likelihood function for each dataset size.
        dataset (list): The observed dataset.
        x_bounds (list): A list of tuples defining the lower and upper bounds of the parameter space for each dimension.
        x_shifts (list): A list of shift vectors to apply during expansion
    """
    def __init__(self, func, args_list, dataset, x_bounds, x_shifts):
        self.func = func
        self.args_list = args_list
        self.x_bounds = x_bounds
        self.x_shifts = x_shifts
        self.n_dim = len(x_bounds)
        self.n_data_points = len(dataset)
        self.grid_results = []
        self.x_spacing_results = []
        self.log_like_results = []
        self.ess_results = []
        self.func_evals = []
        self.grid_size = []
        

    def calc_naive_grid(self, grid_resolution, data_tempering_index, n_processes=4):
        """Exhaustively calculate the log-likelihoods, relative probabilities, weights and effective sample size (ESS) for a given grid resolution.

        Args:
            grid_resolution (int): The resolution of the grid to be evaluated.
            data_tempering_index (int): The index of the current dataset size in args_list.
            n_processes (int): The number of parallel processes to use when evaluating the grid. Defaults to 4.

        Returns:
            tuple: A tuple containing the evaluated grid, spacing, log-likelihoods, relative probabilities, weights, and ESS.
        """
        grid, x_spacing = gt.gen_grid_points(self.n_dim, self.x_bounds, grid_resolution)
        log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, self.func, self.args_list[data_tempering_index], n_processes)
        return grid, x_spacing, log_likelihoods, rel_prob, weights, ess  

    def initialize(self, init_grid_resolution, init_data_size, ess_min, n_processes=4, max_iter=100, store_results=False):
        """Iteratively update the initial grid until the ESS is greater than the specified minimum ESS.

        Args:
            init_grid_resolution (int): The initial resolution of the grid to be evaluated.
            init_data_size (int): The initial size of the dataset.
            ess_min (float): The minimum effective sample size (ESS) to be used for initialization.
            n_processes (int): The number of parallel processes to use when evaluating the grid. Defaults to 4.
            max_iter (int): The maximum number of iterations to use when updating the grid. Defaults to 100.
            store_results (bool, optional): Stores results at each tempering stage (may use a lot of memory for large grids / datasets)
            Defaults to False.

        Returns:
            Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing the following elements:
            - `grid_resolution` (int): The final resolution of the grid.
            - `data_size` (int): The total number of data points used for generating the posterior samples.
            - `grid` (np.ndarray): A 2D numpy array with the coordinates of the grid points.
            - `x_spacing` (np.ndarray): A 1D numpy array with the spacing between the grid points in each dimension.
            - `log_likelihoods` (np.ndarray): A 1D numpy array with the log-likelihood values at each grid point.
            - `rel_prob` (np.ndarray): A 1D numpy array with the relative probabilities of the grid points.
            - `weights` (np.ndarray): A 1D numpy array with the importance weights of the grid points.
            - `ess` (float): The effective sample size (ESS) of the posterior samples.
        """
        # iteratively update initial grid until ess > ess_min
        data_tempering_index = init_data_size - 1  # index offset by 1
        grid, x_spacing, log_likelihoods, rel_prob, weights, ess = self.calc_naive_grid(init_grid_resolution, data_tempering_index, n_processes)

        grid_resolution = init_grid_resolution
        iter = 0
        f_evals = np.size(log_likelihoods)
        with tqdm(total=None, desc="Initializing...") as pbar:     
            while ess < ess_min and iter < max_iter:
                grid_resolution = grid_resolution + 1 
                grid, x_spacing, log_likelihoods, rel_prob, weights, ess = self.calc_naive_grid(grid_resolution, data_tempering_index, n_processes)
                iter = iter + 1
                f_evals = f_evals + np.size(log_likelihoods)
                pbar.update()
                pbar.set_description(f"Intialization: data_size={data_tempering_index+1}, grid_resolution={grid_resolution}, n_grid_points={np.shape(grid)[0]}, ESS={ess}, init_func_evals:{f_evals}")       
        if store_results:
            self.grid_results.append(grid)
            self.x_spacing_results.append(x_spacing)
            self.log_like_results.append(log_likelihoods)
            self.ess_results.append(ess)
            self.func_evals.append(f_evals)
            self.grid_size.append(np.shape(grid)[0])
        return grid_resolution, data_tempering_index+1, grid, x_spacing, log_likelihoods, rel_prob, weights, ess

    def initialize_and_sample(self, init_grid_resolution, init_data_size, ess_min, delta, n_processes=4, max_iter=100, store_results=False):
        """ Initializes the grid and performs the data tempering (iterative batching) to obtain posterior samples.

        Args:
            init_grid_resolution (int): The resolution of the initial grid.
            init_data_size (int): The number of data points used to initialize the grid.
            ess_min (float): The minimum effective sample size (ESS) required for adding a new data point to the grid.
            delta (float): The threshold for removing low probability samples.
            n_processes (int, optional): The number of parallel processes to use for the likelihood evaluations.
                Defaults to 4.
            max_iter (int, optional): The maximum number of iterations for updating the grid.
                Defaults to 100.
            store_results (bool, optional): Stores results at each tempering stage (may use a lot of memory for large grids / datasets)
                Defaults to False.

        Returns:
            Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing the following elements:
            - `grid_resolution` (int): The final resolution of the grid.
            - `data_size` (int): The total number of data points used for generating the posterior samples.
            - `grid` (np.ndarray): A 2D numpy array with the coordinates of the grid points.
            - `x_spacing` (np.ndarray): A 1D numpy array with the spacing between the grid points in each dimension.
            - `log_likelihoods` (np.ndarray): A 1D numpy array with the log-likelihood values at each grid point.
            - `rel_prob` (np.ndarray): A 1D numpy array with the relative probabilities of the grid points.
            - `weights` (np.ndarray): A 1D numpy array with the importance weights of the grid points.
            - `ess` (float): The effective sample size (ESS) of the posterior samples.
        """
        
        # generate initial (uniform) grid samples
        grid_resolution, data_size, grid, x_spacing, log_likelihoods, rel_prob, weights, ess = self.initialize(init_grid_resolution, init_data_size, ess_min, n_processes, max_iter, store_results)

        # iterate through remaining data (i.e. data tempering)
        data_tempering_index = data_size-1
        pbar = tqdm(range(data_tempering_index, self.n_data_points-1),desc='Processing:')
        for i in pbar:
            f_evals = 0
            data_tempering_index = i+1  
            args = self.args_list[data_tempering_index]
            grid = gt.add_grid_points(grid, self.x_bounds, self.x_shifts, x_spacing)  # expand 
            log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, self.func, args, n_processes) 
            f_evals = f_evals + np.size(log_likelihoods)
    
            # ensure ess is high enough for added datapoint, if not, make grid finer
            iter = 0
            while ess<ess_min and iter < max_iter:
                prev_x_spacing = x_spacing
                x_spacing = gt.update_x_spacing(x_spacing,2)  # make grid spacing finer by 2x
                grid = gt.add_grid_points(grid, self.x_bounds, self.x_shifts, x_spacing, prev_x_spacing)  # expand and pack
                log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, self.func, args, n_processes) 
                iter = iter + 1      
                f_evals = f_evals + np.size(log_likelihoods)
     
            grid = gt.reduce_grid_points(grid,weights,delta)

            if store_results:
                self.grid_results.append(grid)
                self.x_spacing_results.append(x_spacing)
                self.log_like_results.append(log_likelihoods)
                self.ess_results.append(ess)
                self.func_evals.append(f_evals)
                self.grid_size.append(np.shape(grid)[0])
    
            pbar.set_description(f"Processing: data_size={data_tempering_index+1}, n_grid_points={np.shape(grid)[0]}, ESS={ess}, func_evals:{f_evals}")
        return  grid_resolution, data_size, grid, x_spacing, log_likelihoods, rel_prob, weights, ess 

   

