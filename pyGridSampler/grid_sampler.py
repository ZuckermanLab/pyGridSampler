from pyGridSampler import grid_tools as gt
import numpy as np
from tqdm import tqdm


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

    def initialize(self, init_grid_resolution, init_data_size, ess_min, n_processes=4, max_iter=100):
        """Iteratively update the initial grid until the ESS is greater than the specified minimum ESS.

        Args:
            init_grid_resolution (int): The initial resolution of the grid to be evaluated.
            init_data_size (int): The initial size of the dataset.
            ess_min (float): The minimum effective sample size (ESS) to be used for initialization.
            n_processes (int): The number of parallel processes to use when evaluating the grid. Defaults to 4.
            max_iter (int): The maximum number of iterations to use when updating the grid. Defaults to 100.

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
        with tqdm(total=None, desc="Initializing...") as pbar:     
            while ess < ess_min and iter < max_iter:
                grid_resolution = grid_resolution + 1 
                grid, x_spacing, log_likelihoods, rel_prob, weights, ess = self.calc_naive_grid(grid_resolution, data_tempering_index, n_processes)
                iter = iter + 1
                pbar.update()
                pbar.set_description(f"Intialization: data_size={data_tempering_index+1}, grid_resolution={grid_resolution}, n_grid_points={np.shape(grid)[0]}, ESS={ess}")
        return grid_resolution, data_tempering_index+1, grid, x_spacing, log_likelihoods, rel_prob, weights, ess

    def initialize_and_sample(self, init_grid_resolution, init_data_size, ess_min, delta, n_processes=4, max_iter=100):
        """ Initializes the grid and performs the data tempering to obtain posterior samples.

        Args:
            init_grid_resolution (int): The resolution of the initial grid.
            init_data_size (int): The number of data points used to initialize the grid.
            ess_min (float): The minimum effective sample size (ESS) required for adding a new data point to the grid.
            delta (float): The threshold for removing low probability samples.
            n_processes (int, optional): The number of parallel processes to use for the likelihood evaluations.
                Defaults to 4.
            max_iter (int, optional): The maximum number of iterations for updating the grid.
                Defaults to 100.

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
        grid_resolution, data_size, grid, x_spacing, log_likelihoods, rel_prob, weights, ess = self.initialize(init_grid_resolution, init_data_size, ess_min, n_processes, max_iter)
        
        # remove low probability samples
        grid = gt.reduce_grid_points(grid,weights,delta) 

        # iterate through remaining data (i.e. data tempering)
        data_tempering_index = data_size-1
        pbar = tqdm(range(data_tempering_index, self.n_data_points-1),desc='Processing:')
        for i in pbar:
            data_tempering_index = i+1  
            args = self.args_list[data_tempering_index]

            #x_spacing = gt.update_x_spacing(x_spacing,2)  # reduce spacing by 2
            grid = gt.add_grid_points(grid, self.x_bounds, self.x_shifts, x_spacing)  # expand and pack
            log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, self.func, args, n_processes) 
    
            # ensure ess is high enough for added datapoint
            iter = 0
            while ess<ess_min and iter < max_iter:
                x_spacing = gt.update_x_spacing(x_spacing,2)  # make grid spacing finer by 2x
                grid = gt.add_grid_points(grid, self.x_bounds, self.x_shifts, x_spacing)  # expand and pack
                log_likelihoods, rel_prob, weights, ess = gt.eval_grid_points(grid, self.func, args, n_processes) 
                iter = iter + 1      
            
            # remove low probability grid points
            grid = gt.reduce_grid_points(grid,weights,delta)
            pbar.set_description(f"Processing: data_size={data_tempering_index+1}, grid_resolution={grid_resolution}, n_grid_points={np.shape(grid)[0]}, ESS={ess}")
        return  grid_resolution, data_size, grid, x_spacing, log_likelihoods, rel_prob, weights, ess 


