r'''
PyGridSampler is a package for adaptive grid-based sampling method using iterative batch sizes.

This method uses multi-grids at different data batch sizes to efficiently evaluate a function surface (e.g. log-likelihood).

The package also comes with utility functions for efficiently creating and evaluating meshgrids.

![grid animation](https://raw.githubusercontent.com/ZuckermanLab/pyGridSampler/main/docs/animation_test.gif)

### Getting started:

install using pip:
```bash
pip install pyGridSampler
```

example usage:
```python
import pygridsampler.grid_sampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(grid_resolution, data_size, ess_min, delta)
```



'''
