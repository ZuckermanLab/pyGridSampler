r'''
PyGridSampler is a package for adaptive grid sampling using data-tempering

example usage:
```python
import GridSampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(init_grid_resolution, init_data_size, ess_min, delta, n_processes=4, max_iter=100)
```

'''