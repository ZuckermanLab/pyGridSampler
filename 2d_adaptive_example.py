import pygrid as g
import agrid as ag
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
from datetime import datetime


def plot_2d_grid(grid_df, title=None):
    '''Helper function to quickly plot grid (i.e. active or expanded grid)'''

    plt.clf()
    x = grid_df['k_1']
    y = grid_df['sigma']
    plt.xlim(0,10)
    plt.ylim(0,1)
    plt.plot(x,y, 'o', markersize=1, color='black')
    plt.ylabel('sigma')
    plt.xlabel('k_1')
    if title:
        plt.title(f'{title}')

    #fname = f'/Users/work/PycharmProjects/pythonProject1/' + f'{title}' + f'.png'
    fname = f'{title}' + f'.png'
    plt.savefig(f'{fname}')


def plot_k_grids(grid_df_list, title=None, labels=None):
    '''Helper function to quickly plot each grid stage at level k'''


    for i,grid_df in enumerate(grid_df_list):

        x = grid_df['k_1']
        y = grid_df['sigma']
        plt.xlim(0, 10)
        plt.ylim(0, 1)
        if labels:
            plt.plot(x, y, 'o', label=labels[i])
            plt.legend()
        else:
            plt.plot(x, y, 'o')

    if title:
        plt.title(f'{title}')
    plt.show()


def generate_synthetic_data(theta):
    '''generates synthetic data for toy model'''
    p = theta
    sigma = theta[-1]
    y_true = model(p)
    y_obs = y_true + np.random.normal(loc=0, scale=sigma, size=np.size(y_true))
    return y_true, y_obs


def model(p):
    ''' toy exponential decay model'''
    k1 = p[0]
    y0 = 50
    t = np.linspace(0, 2, 10)
    y = y0*np.exp(-k1*t)
    return y


##################### 2D Example #########################

t0 = datetime.now()

### CONFIGURATION ###

# set random number generator seed
grid_seed = 1234
np.random.seed(grid_seed)

# generate synthetic data
theta_true = [5, 0.5]
y_true, y_obs = generate_synthetic_data(theta_true)

# define model parameters, boundaries, and initial spacing (number of points per vector)
n_v_points = 2
pv_1 = g.ParameterVector('k_1', 0, 10, n_v_points, pad=1)
pv_2 = g.ParameterVector('sigma', 0, 1, n_v_points, pad=0.1)
p_list = [pv_1, pv_2]
cols=['k_1', 'sigma']  # column names to use for data selection later in the program

# for 2D there are 8 neighboring positions
expansion_list = [[-1,-1],[-1,0],[-1,1],[0,-1], [0,1],[1,-1],[1,0],[1,1]]
packing_list = [[0,1], [1,0], [1,1]]  # only fill in neighbors inside top right 'square'

# set hyper-parameters
ESS_min = 100  # Effective sample size threshold used  for 'good enough' sampling
delta_ESS = 0.001  # threshold used to check how much ESS changes after expansion
delta_w = 0.1  # 1-delta_w determines how much probability to keep during the reduction step
delta_pos = 0.001  # collision = True if distance between point 1 and 2 is less than delta_pos (NOT USING)


### INITIALIZATION ###
n_d_points = 2  # number of data points to include
y_obs_sub = ag.select_data(n_d_points,y_obs)  # subset of full dataset
grid_level = 0  # indexes how many density 'levels' have been added since the initial grid
ESS = 0  # effective sample size
n_iter = len(y_obs) - 1  # iterate the main loop over each datapoint, but the first iteration includes 2 data points

# create initial grid
grid = g.MeshGrid(p_list)  # creates Grid object
_, _ = grid.mesh_grid(grid.p_vectors)  # create temporary mesh grid then randomize

### Fix this!
# try different random shift amounts until they are within boundary
b_err = True
i = 0
while b_err is True:
    try:
        mg, mg_df = grid.randomize_grid(seed=grid_seed+i)  # randomize grid
    except:
        b_err = True
        i = i + 1
    else:
        b_err = False
        grid_seed = grid_seed + i  # keep new seed for future randomization calls

# FOR DEBUGGING/TESTING ONLY - store grid stages at each grid 'level' - i.e. with each added data point
active_grid_list = []
reduced_grid_list = []
expanded_grid_list = []
packed_grid_list = []


### MAIN LOOP
for i in range(n_iter):  # i --> number of datapoints to include + 2 (using both endpoints at first iteration)
    print(i)  # for debugging
    if i == 0: # for initial iteration
        expansion_spacing = 0  # track the spacing for expansion step
        packing_spacing = 0  # also for packing step
        while ESS < ESS_min:
            n_v_points = n_v_points + 1  # increase number of vector points
            pv_1.change_density(n_v_points)
            pv_2.change_density(n_v_points)
            grid = g.MeshGrid(p_list)
            _, _ = grid.mesh_grid(grid.p_vectors)  # create initial grid with higher density
            mg_i, mg_df_i = grid.randomize_grid(seed=grid_seed)  # randomize grid
            logp_list = []  # store logl calculations
            ID_list = []  # store IDs
            for j, row in mg_df_i.iterrows():  # loop over each grid point
                theta_j = mg_df_i.iloc[j]  # grid set (parameter set) at index i
                y_pred_j = model(theta_j)
                y_pred_j_sub = ag.select_data(n_d_points, y_pred_j)
                sigma_j = mg_df_i.iloc[j][
                    'sigma']  # store sigma_j to use in logl calc <-- HARDCODED 'sigma' LABEL FOR NOW
                logl_temp = g.calc_logl(y_pred_j_sub, y_obs_sub, sigma_j)  # calculate logl for the subset of y_obs
                logp_list.append(logl_temp)
                ID_j = ag.generate_ID(theta_j,cols=['k_1', 'sigma'])  # generate unique ID for each grid point - used for collisions checks
                ID_list.append(ID_j)

            mg_df_i['logp'] = logp_list  # update dataframe to include logl <-- HARDCODED 'logp' LABEL FOR NOW
            rel_logp = g.calc_rel_logp(logp_list)  # calculate relative log probability
            rel_p = g.calc_rel_p(rel_logp)  # calculate relative probability
            mg_df_i['w'] = g.calc_norm_weights(rel_p)  # normalized weights (sum p = 1) <-- HARDCODED 'w' LABEL FOR NOW
            ESS = g.calc_ess(rel_p)  # calculate the effective sample size
            mg_df_i['ID'] = ID_list  # FIX THIS! set initial IDs to index <-- HARDCODED 'ID' LABEL FOR NOW
            spacing = np.array(
                [pv.spacing for pv in grid.p_list])  # spacing for each n parameter is saved as an nx1 array
            expansion_spacing = np.array([pv.spacing/(2**(0)) for pv in grid.p_list])
            packing_spacing = np.array([pv.spacing / (2 ** (1)) for pv in grid.p_list])
        print(f'ESS: {ESS}')

        # store results from current grid which has ESS > ESS_min for 2 data points
        active_grid_list.append(mg_df_i)
        reduced_grid_list.append(ag.reduce_grid(active_grid_list[i], delta=delta_w))
        grid_ID_set = ag.generate_set(reduced_grid_list[i]['ID'])
        expansion_points = ag.expand_grid(reduced_grid_list[i],
                                                 grid_ID_set,
                                                 expansion_spacing,
                                                 x_list=expansion_list,
                                                 cols=cols)
        expanded_grid_list.append(pd.concat([reduced_grid_list[i],expansion_points]))
        packed_points = ag.pack_grid(expanded_grid_list[i],
                                          packing_spacing,
                                          x_list=packing_list,
                                          cols=cols)
        packed_grid_list.append(pd.concat([expanded_grid_list[i],packed_points]))

        plot_2d_grid(active_grid_list[i], title=f'active grid, k={grid_level}')
        plot_2d_grid(reduced_grid_list[i], title=f'reduced grid, k={grid_level}')
        plot_2d_grid(expanded_grid_list[i], title=f'expanded grid, k={grid_level}')
        plot_2d_grid(packed_grid_list[i], title=f'packed grid, k={grid_level}')

    else:  # every iteration that is not the initial grid
        # note: one challenge is keeping track of the right grids, indices, lists, spacing, and levels. This should be refactored and tested!

        # increase number of points and grid level
        n_d_points = n_d_points + 1
        grid_level = i + 1
        grid_i = packed_grid_list[i-1]  # current 'active' grid is the previously 'packed' grid
        print(f'{len(grid_i)}')
        expansion_spacing = np.array([pv.spacing / (2 ** (i)) for pv in grid.p_list])  # expand out at 2^(n-1)
        packed_spacing_0 = np.array([pv.spacing / (2 ** (grid_level)) for pv in grid.p_list])
        plot_2d_grid(grid_i, title=f'test{i}')

        # check calculate new ESS with data point added
        logp_list = []  # store logl calculations
        ID_list = []  # store IDs
        for j, row in grid_i.iterrows():  # loop over each grid point
            theta_j = row[cols]  # grid set (parameter set) at index i
            y_pred_j = model(theta_j)
            y_pred_j_sub = ag.select_data(n_d_points, y_pred_j)
            y_obs_sub = ag.select_data(n_d_points, y_obs)  # subset of full dataset
            sigma_j = grid_i.iloc[j][
                'sigma']  # store sigma_j to use in logl calc <-- HARDCODED 'sigma' LABEL FOR NOW
            logl_temp = g.calc_logl(y_pred_j_sub, y_obs_sub, sigma_j)  # calculate logl for the subset of y_obs
            logp_list.append(logl_temp)
            ID_j = ag.generate_ID(theta_j,cols=['k_1', 'sigma'])  # generate unique ID for each grid point - used for collisions checks
            ID_list.append(ID_j)
        grid_i['logp'] = logp_list  # update dataframe to include logl <-- HARDCODED 'logp' LABEL FOR NOW
        rel_logp = g.calc_rel_logp(logp_list)  # calculate relative log probability
        rel_p = g.calc_rel_p(rel_logp)  # calculate relative probability
        grid_i['w'] = g.calc_norm_weights(rel_p)  # normalized weights (sum p = 1) <-- HARDCODED 'w' LABEL FOR NOW
        ESS = g.calc_ess(rel_p)  # calculate the effective sample size
        grid_i['ID'] = ID_list  # FIX THIS! set initial IDs to index <-- HARDCODED 'ID' LABEL FOR NOW

        print(f'ESS: {ESS}')

        # check if new grid has high enough ESS (with data point added)
        # this follows the same logic as above
        # FIX! should write this as a function so I don't repeat it
        if ESS > ESS_min:

            # store results from current grid which has ESS > ESS_min for x data points
            active_grid_list.append(grid_i)
            reduced_grid_list.append(ag.reduce_grid(active_grid_list[i], delta=delta_w))

            grid_ID_set = ag.generate_set(reduced_grid_list[i]['ID'])
            expansion_spacing = np.array([pv.spacing / (2 ** (grid_level - 2)) for pv in grid.p_list])
            expansion_points = ag.expand_grid(reduced_grid_list[i],
                                              grid_ID_set,
                                              expansion_spacing,
                                              x_list=expansion_list,
                                              cols=cols)
            expanded_grid_list.append(pd.concat([reduced_grid_list[i], expansion_points]))
            packed_spacing = np.array([pv.spacing / (2 ** (grid_level - 1)) for pv in grid.p_list])
            packed_points = ag.pack_grid(expanded_grid_list[i],
                                         packed_spacing_0,
                                         x_list=packing_list,
                                         cols=cols)
            packed_grid_list.append(pd.concat([expanded_grid_list[i], packed_points]))

            # plot
            plot_2d_grid(active_grid_list[i], title=f'active grid, k={i}')
            plot_2d_grid(reduced_grid_list[i], title=f'reduced grid, k={i}')
            plot_2d_grid(expanded_grid_list[i], title=f'expanded grid, k={i}')
            plot_2d_grid(packed_grid_list[i], title=f'packed grid, k={i}')

        else:
            # FIX - same logic as above, should write a function to not repeat the code
            while ESS < ESS_min:
                grid_level = grid_level + 1
                packing_spacing = np.array([pv.spacing / (2 ** (grid_level-2)) for pv in grid.p_list])
                packed_points_i_temp = ag.pack_grid(grid_i,
                                             packing_spacing,
                                             x_list=packing_list,
                                             cols=cols)
                packed_points_i = pd.concat([packed_points_i_temp,grid_i])

                # check ESS
                logp_list = []  # store logl calculations
                ID_list = []  # store IDs
                for j, row in packed_points_i.iterrows():  # loop over each grid point
                    theta_j = row[cols]  # grid set (parameter set) at index i
                    y_pred_j = model(theta_j)
                    y_pred_j_sub = ag.select_data(n_d_points, y_pred_j)
                    y_obs_sub = ag.select_data(n_d_points, y_obs)  # subset of full dataset
                    sigma_j = packed_points_i.iloc[j][
                        'sigma']  # store sigma_j to use in logl calc <-- HARDCODED 'sigma' LABEL FOR NOW
                    logl_temp = g.calc_logl(y_pred_j_sub, y_obs_sub, sigma_j)  # calculate logl for the subset of y_obs
                    logp_list.append(logl_temp)
                    ID_j = ag.generate_ID(theta_j,cols=['k_1', 'sigma'])  # generate unique ID for each grid point - used for collisions checks
                    ID_list.append(ID_j)
                packed_points_i['logp'] = logp_list  # update dataframe to include logl <-- HARDCODED 'logp' LABEL FOR NOW
                rel_logp = g.calc_rel_logp(logp_list)  # calculate relative log probability
                rel_p = g.calc_rel_p(rel_logp)  # calculate relative probability
                packed_points_i['w'] = g.calc_norm_weights(rel_p)  # normalized weights (sum p = 1) <-- HARDCODED 'w' LABEL FOR NOW
                ESS = g.calc_ess(rel_p)  # calculate the effective sample size
                packed_points_i['ID'] = ID_list  # FIX THIS! set initial IDs to index <-- HARDCODED 'ID' LABEL FOR NOW
                print(f'ESS: {ESS}')

            # store results from current grid which has ESS > ESS_min for x data points
            active_grid_list.append(packed_points_i)
            reduced_grid_list.append(ag.reduce_grid(active_grid_list[i], delta=delta_w))
            grid_ID_set = ag.generate_set(reduced_grid_list[i]['ID'])
            expansion_spacing = np.array([pv.spacing / (2 ** (grid_level - 2)) for pv in grid.p_list])
            expansion_points = ag.expand_grid(reduced_grid_list[i],
                                              grid_ID_set,
                                              expansion_spacing,
                                              x_list=expansion_list,
                                              cols=cols)
            expanded_grid_list.append(pd.concat([reduced_grid_list[i], expansion_points]))
            packed_spacing = np.array([pv.spacing / (2 ** (grid_level - 1)) for pv in grid.p_list])
            packed_points = ag.pack_grid(expanded_grid_list[i],
                                         packed_spacing,
                                         x_list=packing_list,
                                         cols=cols)
            packed_grid_list.append(pd.concat([expanded_grid_list[i], packed_points]))

            # plot
            plot_2d_grid(active_grid_list[i], title=f'active grid, k={i}')
            plot_2d_grid(reduced_grid_list[i], title=f'reduced grid, k={i}')
            plot_2d_grid(expanded_grid_list[i], title=f'expanded grid, k={i}')
            plot_2d_grid(packed_grid_list[i], title=f'packed grid, k={i}')

print(datetime.now()-t0)  # runtime