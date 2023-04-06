r'''
PyGridSampler is a package for adaptive grid sampling using data-tempering

![Alt Text](https://github.com/ZuckermanLab/pyGridSampler/main/docs/animation_test.gif)

example usage:
```python
import GridSampler as gs

sampler = gs.GridSampler(log_prob_func, func_args, data, x_bounds, x_extension)
results = sampler.initialize_and_sample(grid_resolution, data_size, ess_min, delta)
```



'''