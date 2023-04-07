import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def get_marginals_from_grid(grid):
    """Transposes the input grid to obtain a list of marginal distributions.

    Args:
        grid (np.ndarray): The input grid.

    Returns:
        list: A list of marginal distributions.
    """
    grid_marginals = np.transpose(grid).tolist()
    return grid_marginals


def plot_grid_marginals(grid, x_bounds, truths, fname):
    """Plots the 1D marginal distributions of the input grid.

    Args:
        grid (np.ndarray): The input grid.
        x_bounds (list): A list of x bounds for each dimension of the grid.
        truths (list): A list of true values for each dimension of the grid.
        fname (str): The filename of the output plot.
    """
    marginals = get_marginals_from_grid(grid)
    fig, axs = plt.subplots(1, len(marginals), figsize=(10, 5))
    for i in range(len(marginals)):
        if truths:
            axs[i].hist(marginals[i], density=True, bins=50, range=x_bounds[i], histtype='step')
            axs[i].axvline(truths[i], color='black', ls='--')
        else:
            axs[i].hist(marginals[i], density=True, bins=50, range=x_bounds[i], histtype='step' )
        axs[i].set_xlabel(f'x_{i+1}')
        axs[i].set_ylabel('density')
    plt.suptitle('grid 1D marginal distributions')
    plt.tight_layout()
    plt.savefig(fname)


def eval_grid_points(grid, func, args, n_processes=4):
    """
    Evaluates the given function over the provided grid points in parallel.

    Args:
        grid (np.ndarray): A numpy array of shape (n_points, n_dim) representing the grid points to evaluate the function on.
        func (callable): A function to evaluate on the grid points, f(X, args).
        args (tuple): A tuple of additional arguments to pass to the function.
        n_processes (int): The number of processes to use for parallel evaluation.

    Returns:
        A tuple containing:
        - log_likelihoods (np.ndarray): A numpy array of shape (n_points,) containing the log-likelihood of each grid point.
        - rel_prob (np.ndarray): A numpy array of shape (n_points,) containing the relative probability (likelihood) of each grid point.
        - weights (np.ndarray): A numpy array of shape (n_points,) containing the weights used for resampling.
        - ess (float): The effective sample size (ESS) of the grid points.
    """
    with Pool(processes=n_processes) as pool:
        log_likelihoods = pool.starmap(func, [(X, args) for X in grid])
    rel_prob = calc_rel_p(calc_rel_logp(log_likelihoods))
    weights = calc_norm_weights(rel_prob)
    ess = calc_ess(rel_prob)
    return log_likelihoods, rel_prob, weights, ess


def gen_grid_points(n_dim, x_bounds, grid_resolution): 
    """Generate a grid of evenly spaced points within the specified bounds.

    Args:
        n_dim (int): The number of dimensions of the grid.
        x_bounds (List[Tuple[float, float]]): A list of tuples, where each tuple contains the lower and upper bounds of the corresponding dimension.
        grid_resolution (int): The number of points along each dimension.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing the grid of points (as a numpy array with shape (n_points, n_dim)) and the spacing between the points along each dimension.
    """
    x_points = [np.linspace(bound[0], bound[1], grid_resolution) for bound in x_bounds] 
    x_spacing = [(bound[1] - bound[0]) / (grid_resolution - 1) for bound in x_bounds]
    initial_grid = np.array(np.meshgrid(*x_points)).T.reshape(-1,n_dim)
    return initial_grid, x_spacing


def update_x_spacing(x_spacing, div_amount):
    """
    Update spacing between grid points by dividing the initial spacing by a specified amount.

    Args:
    - x_spacing (list): A list of floats representing the initial spacing between grid points.
    - div_amount (float): A float specifying the amount to divide the initial spacing by.

    Returns:
    - new_x_spacing (list): A list of floats representing the updated spacing between grid points.
    """
    new_x_spacing = [x/div_amount for x in x_spacing]
    return new_x_spacing


def reduce_grid_points(grid, weights, delta):
    """
    Reduce the number of grid points by keeping only the points with the highest weights until a certain proportion of the total weight is reached.

    Args:
    - grid (np.ndarray): A numpy array representing the initial grid of points to be reduced.
    - weights (np.ndarray): A numpy array representing the weight of each point on the grid.
    - delta (float): A float specifying the proportion of the total weight to be kept.

    Returns:
    - reduced_grid (np.ndarray): A numpy array representing the reduced grid of points.
    """
    weights_sorted_idx = np.argsort(weights)[::-1]
    weights_sorted = weights[weights_sorted_idx]
    cumulative_sum = np.cumsum(weights_sorted)
    threshold_index = np.argwhere(cumulative_sum >= 1-delta)[0, 0]
    indices_to_keep = weights_sorted_idx[:threshold_index+1]
    reduced_grid = grid[indices_to_keep]  
    return reduced_grid


def check_grid_boundary(grid, x_bounds):
    """
    Check if any points on the grid are outside the specified boundaries and remove them.
    Time complexity of ~ O(n_dim*n_array_elements)

    Args:
    - grid (np.ndarray): A numpy array representing the grid of points to be checked.
    - x_bounds (list): A list of tuples representing the lower and upper bounds for each dimension.

    Returns:
    - updated_grid (np.ndarray): A numpy array representing the updated grid of points with points outside the boundaries removed.
    """
    n_dim = grid.shape[1]
    boundary_mask = np.ones(grid.shape[0], dtype=bool)  # initial mask = True for all array elements
    for i in range(n_dim):
        lb, ub = x_bounds[i]
        boundary_mask &= ((grid[:, i] >= lb) & (grid[:, i] <= ub))  # update mask for each dimension and element
    updated_grid = grid[boundary_mask]
    return updated_grid


def add_grid_points(grid, x_bounds, x_shifts, x_spacing, prev_x_spacing=None):
    """
    Add new grid points to the existing grid based on spacing, shifts, and boundaries.

    This function expands the grid by adding new grid points shifted by x_shifts and
    x_spacing, making sure that the new points are within the x_bounds and not too close
    to existing points or other new points being added.

    Args:
        grid (array-like): The initial grid of points (n_points, n_dim).
        x_bounds (array-like): The lower and upper bounds for each dimension (n_dim, 2).
        x_shifts (array-like): The possible shift directions (n_shifts, n_dim).
        x_spacing (array-like): The spacing in each dimension (n_dim,).
        prev_x_spacing (array-like, optional): The previous grid spacing in each dimension (n_dim,).
            Default is None.

    Returns:
        array-like: The expanded grid with new points added (n_points + n_added_points, n_dim).
    """
    kdtree = KDTree(grid)
    if prev_x_spacing is not None and prev_x_spacing == x_spacing:
        min_dist_fraction = 0.5
        min_dist = min_dist_fraction * min(x_spacing)
    else:
        min_dist_fraction = 0.25
        min_dist = min_dist_fraction * min(x_spacing)
    expanded_points = []
    added_points = []
    for point in grid:
        for shift in x_shifts:
            shifted_point = point + np.array(shift) * np.array(x_spacing)
            
            if not check_grid_point_boundary(shifted_point, x_bounds):
                continue
            
            dist, _ = kdtree.query(shifted_point)
            if dist > min_dist:
                if len(added_points) > 0:
                    added_tree = KDTree(added_points)
                    added_dist, _ = added_tree.query(shifted_point)
                    if added_dist < min_dist:
                        continue
                
                expanded_points.append(tuple(shifted_point))
                added_points.append(shifted_point)

    if len(expanded_points) == 0:
        expanded_grid = grid
    else:
        expanded_points = np.array(expanded_points)
        expanded_grid = np.vstack((grid, expanded_points))
    return expanded_grid


def check_grid_point_boundary(point, x_bounds):
    """
    Check if a given point is within the specified boundaries.

    Args:
        point (array-like): The point to be checked (n_dim,).
        x_bounds (array-like): The lower and upper bounds for each dimension (n_dim, 2).

    Returns:
        bool: True if the point is within the bounds, False otherwise.
    """
    n_dim = len(x_bounds)
    in_bounds = True
    for i in range(n_dim):
        lb, ub = x_bounds[i]
        in_bounds &= (point[i] >= lb) & (point[i] <= ub)
    return in_bounds


def calc_rel_logp(logp):
    """Calculate relative log probability (log-likelihood).
    
    Args:
        logp (numpy.ndarray): Array of log probabilities.
    
    Returns:
        numpy.ndarray: Array of relative log probabilities.
            log(p/pmax) = log(p) - ln(p_max).
    
    Note: 
        log(p) --> log(p/pmax) = log(p) - ln(p_max)
    """
    return logp - np.max(logp)


def calc_rel_p(rel_logp):
    """Calculate relative probability (likelihood).
    
    Args:
        rel_logp (numpy.ndarray): Array of relative log probabilities.
    
    Returns:
        numpy.ndarray: Array of relative probabilities.
           
    Note:
        rel_p = p/pmax = e**log(p/pmax)
    """
    return np.exp(rel_logp)


def calc_norm_weights(p):
    """Calculate normalized weights used for resampling.
    
    Args:
        p (numpy.ndarray): Array of probabilities.
    
    Returns:
        numpy.ndarray: Array of normalized weights.

    Note:
        w = p/sum(p)
    """
    return p / np.sum(p)


def calc_ess(rel_p):
    """Calculate effective sample size (ESS).
    
    Args:
        rel_p (numpy.ndarray): Array of relative probabilities.
    
    Returns:
        float: Effective sample size (ESS).
    
    Note:
        ESS = sum(p/pmax) = sum(relative p).
    """
    return np.sum(rel_p)


def get_sorted_idx_sublists(data):
    """
    Splits a list of data into sublists by index order.

    Args:
        data (list): A list of data.

    Returns:
        list: A list of sublists in ascending length, in index order.

    Example:
        >>> data = [1, 2, 3]
        >>> get_sorted_idx_sublists(data)
        [[0], [0, 1], [0, 1, 2]]

    """
    idx_list = list(range(len(data)))
    idx_sublist = [idx_list[:i+1] for i in range(len(idx_list))] 
    return idx_sublist
    

def get_sorted_sublists_from_idx(idx_sublist, x):
    """
    Given a list of indices and a list x, return a sorted list of sublists from x.

    Args:
        idx_sublist (list): A list of sublists sorted by ascending length, in index order.
        x (list): A list of data.

    Returns:
        list: A list of sublists of the data, sorted by increasing length.

    Example:
        >>> x = [1, 2, 3]
        >>> idx_sublist = [[0], [0, 1], [0, 1, 2]]
        >>> get_sorted_sublists_from_idx(idx_sublist, x)
        [[1], [1, 2], [1, 2, 3]]
    """
    x_sublist = [x[i] for i in idx_sublist]
    return x_sublist

def plot_2d_scatter(grid_xy, title, x_label, y_label, x_lim, y_lim, x_true, y_true, fname=[]):
    """Plots a 2D scatter plot of the grid results.

    Args:
        grid_xy (List[Tuple[float, float]]): The coordinates of the grid points.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        x_lim (Tuple[float, float]): The limits for the x-axis.
        y_lim (Tuple[float, float]): The limits for the y-axis.
        x_true (float): The true x value.
        y_true (float): The true y value.
        fname (str): The file name to save the plot as. Default is empty (not saved)

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    fig = plt.figure(figsize=(10,7))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    y = [x[1] for x in grid_xy]
    x = [x[0] for x in grid_xy]
    plt.scatter(x,y, alpha=0.45, s=1, color='black')
    plt.axvline(x=x_true, color='black', linestyle='--')
    plt.axhline(y=y_true, color='black', linestyle='--')
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    return fig

