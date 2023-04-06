# Grid sampling with python
[![Tests](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/python-package.yml/badge.svg)](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/python-package.yml)

Evaluate log-likelihood and log-posterior surfaces using an adaptive grid and iterative batch sizes. 

Getting started:

```
import GridSampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(grid_resolution, data_size, ess_min, delta)
```
see example notebook and [Docs](https://zuckermanlab.github.io/pyGridSampler/pyGridSampler.html).


Preprint (please cite):

*August George, Zuckerman Lab, OHSU, 2023*
