# Grid sampling with python

Evaluate log-likelihood and log-posterior surfaces using an adaptive grid and iterative batch sizes. 

Getting started:

```
import GridSampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(grid_resolution, data_size, ess_min, delta)
```
see example notebook and [docs].


Preprint (please cite):

*August George, Zuckerman Lab, OHSU, 2023*
