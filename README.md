# Grid sampling with python
[![Tests](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/python-package.yml/badge.svg)](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/python-package.yml)
[![Docs](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/setup-docs.yml/badge.svg)](https://github.com/ZuckermanLab/pyGridSampler/actions/workflows/setup-docs.yml)

Evaluate log-likelihood and log-posterior surfaces using an adaptive multigrid and iterative batch sizes. 


Getting started:

```
import GridSampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(grid_resolution, data_size, ess_min, delta)
```
see [Docs](https://zuckermanlab.github.io/pyGridSampler/pyGridSampler.html) and included example notebook.


Preprint (please cite):

*August George, Zuckerman Lab, OHSU, 2023*
