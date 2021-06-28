import pygrid as g
import numpy as np
import datetime
import logging
import os
import ray


# output configuration
#date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
date = '1234'
cwd_path = os.getcwd()
f_name = os.path.join(cwd_path,date)
logging.basicConfig(filename=f'{f_name}_log.txt', level=logging.INFO)
logging.info(f'output path: {f_name}')


# define model
def model(p):
    ''' toy exponential decay model'''
    k1 = p[0]
    k2 = p[1]
    y0 = 1
    t = np.linspace(0, 10, 100)
    y = k1*np.exp(-(k2)*t)
    return y


# convenience function for printing
def print_parameter_vectors(p_list):
    ''' convenience function to log the parameter name, min, max, etc'''
    logging.info(f'using {len(p_list)} parameters')
    for p in p_list:
        logging.info(f' name: {p.name}')
        logging.info(f' min: {p.min}')
        logging.info(f' max: {p.max}')
        logging.info(f' n points: {p.div}')
        logging.info(f' vector: {p.vector}')
        logging.info(f' spacing: {p.spacing}\n')




# generate synthetic data
logging.info(f'generating synthetic data')
seed = 1234
p_true = [2,8,0.5]
y_true = model(p_true[:2])
y_obs = y_true + np.random.normal(loc=0, scale=p_true[-1], size=np.size(y_true))
logging.info(f' synthetic data seed: {seed}')
logging.info(f' p_true: {p_true}')
logging.info(f' y_true: {y_true}')
logging.info(f' y_obs: {y_obs}\n')

# generate parameter vectors
logging.info(f'generating parameter vectors')
n_p_div = 5
p1 = g.ParameterVector('k_1', 0, 4, n_p_div, 1)  # add padding
p2 = g.ParameterVector('k_2', 6, 10, n_p_div, 1)
p3 = g.ParameterVector('sigma', 0.1, 1, n_p_div, 0.1)
p_list = [p1, p2, p3]
print_parameter_vectors(p_list)

# generate mesh grid
logging.info(f'generating mesh grid')
grid = g.MeshGrid(p_list)
_, _ = grid.mesh_grid(grid.p_vectors)  # create initial grid
grid_seed = 1234
logging.info(f' randomizing grid using seed {grid_seed}')


# FIX this! - find a better way to check boundary
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

#mg, mg_df = grid.randomize_grid(seed=grid_seed)
logging.info(f' randomized parameter values:')
print_parameter_vectors(grid.p_list)  # different now from initial parameter vectors
logging.info(f' max logl observed, logl(x=y_obs | mu=y_obs, sigma=sigma_true): {g.calc_logl(y_obs,y_obs,p_true[-1])}\n')

# settings to calculate log-likelihood
n_resamples = 1000
logging.info(f'n resamples: {n_resamples}')

# used to run ray remote function - this runs in parallel
@ray.remote
def f(i, df):
    """ This calculates the log likelihood based on the observed data and theta"""
    y_pred_i = model(df.iloc[i])
    sigma_i = df.iloc[i]['sigma']
    return g.calc_logl(y_pred_i,y_obs,sigma_i)

# initialize and run Ray remote function to calculate log-likelihood
logging.info('initializing Ray')
ray.init()
logp_b_ref = []
mg_id = ray.put(mg_df)
for j in range(len(mg_df.index)):
    logp_b_ref.append(f.remote(j, mg_id))  # list (in the same order as meh grid index)
logging.info('calculating log-likelihood')
logp = ray.get(logp_b_ref)  # run ray remote functions

# update dataframe with logl, calculate relative likelihood, relative probability, normalized weight, ESS
mg_df['logp'] = logp
rel_logp = g.calc_rel_logp(logp)
rel_p = g.calc_rel_p(rel_logp)
norm_w = g.calc_norm_weights(rel_p)
ess = g.calc_ess(rel_p)
logging.info(f' df: {mg_df}')
mg_df.to_csv(f'{f_name}_df.csv')
logging.info(f' rel logp: {rel_logp}')
logging.info(f' rel p: {rel_p }')
logging.info(f' norm weights: {norm_w}')
logging.info(f' ESS: {ess}')

# resample
logging.info(f'resampling')
resampled_points = g.resample(mg_df,n_resamples,norm_w)
logging.info(f' resampled points: {resampled_points}')
resampled_points.to_csv(f'{f_name}_resample.csv')
