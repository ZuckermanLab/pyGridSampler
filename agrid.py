import numpy as np
import pandas as pd


def select_data(m, y_obs):
    '''Convenience function which creates an ordered list of m data points from the full dataset
       :param m: an integer that determines how many datapoints to include
       :param y_obs: the full observed dataset
       Note: Data points are returned in a list ordered first, last, then in order from second to second to last
       :return: y: a subset of the full data set containing m points
    '''
    y=[y_obs[0],y_obs[-1]]
    if m > 2:
        for i in range(m-2):
            y.append(y_obs[1+i])
    return y


def reduce_grid(grid_df, delta):
    '''
    Reduces an 'active' grid to a grid with the highest probability points - as determined by a delta hyper parameter
    :param grid_df: a dataframe containing the 'active' grid
    :param delta: a float that determines how much probability is being kept in the 'reduced' grid.
    Note: the smaller delta is, the more parameter sets will be kept, and the less the active grid will be reduced
    :return: temp_df: a dataframe containing the reduced grid of highly probable models
    '''
    sorted_df = grid_df.sort_values(by=['w'], ascending=False)  # store the sorted active dataframe
    i = 0
    temp_df = sorted_df.iloc[0:i]  # initialize a temporary dataframe to store the highest probability parameter sets
    temp_sum = temp_df['w'].sum()  # initialize the sum of the currently stored high probability parameter sets
    while temp_sum < 1-delta:  # while the total stored probability is less than the threshold (1-delta)
        i = i + 1
        temp_df = sorted_df.iloc[0:i]  # include another parameter set
        temp_sum = temp_df['w'].sum()  # sum the probability of the selected parameter sets
    return temp_df


def expand_grid(grid_df, grid_ID_set, spacing, x_list, cols):
    '''
    Expands the current grid points to include their neighbors
    :param grid_df: a dataframe containing the grid points
    :param grid_ID_set: a Set of grid IDs to check for collisions
    :param spacing: the spacing distance between grid points, for each vector
    :param x_list: a list of expansion moves for each neighbor - e.g. 2D has 8 neighbors, for x_list has 8 elements,
    ([-1,-1]...[0,0]...[1,1])
    :param cols: the parameter names used to select from the grid database columns
    :return: a dataframe containing the new, non-colliding grid points and their IDs
    '''
    new_point_list = []
    new_ID_list = []
    new_ID_set = set()
    # go through each point in active grid
    for j, row in grid_df.iterrows():
        temp_grid_point = row[cols]
        for x in x_list:
            new_point = temp_grid_point + x*spacing
            new_ID = generate_ID(new_point, cols)
            if new_ID not in grid_ID_set and new_ID not in new_ID_set:  # if there is no collision
                # FIX! - still need to check boundary conditions!
                new_point_list.append(new_point)  # add new point to list
                new_ID_list.append(new_ID)  # add new point ID to list
                new_ID_set = set(new_ID_list)  # keep newly added points for collision detected
    temp_df = pd.DataFrame(new_point_list)  # create dataframe for the new points
    temp_df['ID'] = new_ID_list  # add ID label column
    return temp_df


def pack_grid(grid_df, spacing, x_list, cols):
    '''
    Packs more grid points in between the existing ones to increase the grid density
    :param grid_df: a dataframe containing the grid points
    :param spacing: the spacing distance between grid points, for each vector
    :param x_list: a list of expansion moves for each neighbor within the upper right box- e.g. 2D has 8 neighbors, for x_list has 8 elements,
    ([-1,-1]...[0,0]...[1,1])
    :param cols: the parameter names used to select from the grid database columns
    :return: a dataframe containing the new, non-colliding grid points and their IDs
    note: x_list is different than the expansion step x_list to reduce redundant calculations
            e.g. for 2D only 3 neighbors are used [0,1], [1,0], [1,1]
    note 2: collision checks are not necessary since the new points should be inside the boundaries
    '''
    new_point_list = []
    new_ID_list = []
    # go through each point in active grid
    for j , row in grid_df.iterrows():
        temp_grid_point = row[cols]
        for x in x_list:
            new_point = temp_grid_point + x*spacing
            new_ID = generate_ID(new_point,cols)
            new_point_list.append(new_point)  # add new point to list
            new_ID_list.append(new_ID)  # add new point ID to list

    temp_df = pd.DataFrame(new_point_list)  # create dataframe for the new points
    temp_df['ID'] = new_ID_list  # add ID label column
    return temp_df


def generate_ID(theta, cols):
    ''' Create a tuple of the parameter set (theta) based on the selected columns'''
    rounded_id = theta[cols].astype(np.double).round(12)  # convert to float datatype and round to 15 decimal places
    return tuple(rounded_id)  # round to 15 decimal places


def generate_set(temp_list):
    ''' Creates a set from the IDs in the grid '''
    return set(temp_list)



def encode_name(df, level):
    '''EMPTY - retry alphabetical naming scheme'''
    pass

def decode_name(df, name):
    '''EMPTY - retry alphabetical naming scheme'''
    pass

