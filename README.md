# GridSampler README - Grid sampling with python
This repository contains a collection of python scripts and examples related to various mesh grid sampling methods. All code is fairly well documented but is under development and has had limited testing - procede with caution! See the development document for more details on the current development. 

## Grid sampler module - grid.py
This module contains two custom classes as well as many helper functions to aid in the creation and manipulation of mesh grids. In short, the full parameter space is discretized into a set of grid points which serve as seed points for future sampling.  Support for batching and HPC is in progress.

Custom classes: 
1. Parameter Vector - creates a linearly spaced vector w/ name, min, max, and number of points with or without padding. Can update the vector density by adding more points, and can randomize the point values by a constant value. 
2. Mesh Grid - creates a grid object that can create a mesh grid from a list of Parameter Vector objects. The mesh grid can be updated and randomized based on inputs. 

Helper functions:
* A collection of functions that calculate log-likelihood, relative log-likelihood, relative probability, normalized weights (probability), effective sample size, and resampling. In development are support for multiple batches - calculating batch probability and adjusted effective sample size.

Usage: 
* Create a few parameter vector objects and use those to create a grid object. From there you can do a variety of tasks depending on the type of sampling required - see included single batch example 

## Grid sampler example - single_batch_example.py
Overview: This example script generates and resamples from a simple 2d exponential decay model using the grid sampler module. First the model f(theta) is defined and synthetic data is generated with Gaussian noise. A bounded parameter space is created from domain knowledge and is turned into a discrete randomized mesh grid determined by user set hyperparameters. From this mesh grid the log-likelihood and estimated effective sample size are calculated. The mesh grid is resampled N times (based on calculated weights) - these resampled points can then be ran in a traditional MCMC routine. This example also demonstrates logging the data and using Ray for simple parallelization.

## [Under heavy development] Adaptive grid sampler module - agrid.py 
This module builds off of the grid sampler module described above. It contains several helper functions that aid in an 'adaptive data mesh grid' procedure. This procedure borrows from adaptive data tempering/annealed importance sampling/secquential Monte Carlo methods. Briefly, the idea is to start with a small number of datapoints so that the log-likelihood distribution is initially very broad and flat. The grid then reduces down to the top N likeliest grid points. These points are then expanded to the nearest neighbors and 'packed' with in-between points, increasing the grid point density in the region near the likeliest points. Another data point is added and this process is repreated until all datapoints have been added. 

Helper functions:
* a collection of functions that order and select the amount of data, reduces the number of grid points, expands the number of grid points, 'packs' the grid, generates and sets a unique gridpoint ID.
    
Usage:
* similar to the mesh grid described above. Create a few parameter vector objects and use those to create a grid object. From there you can do a variety of tasks depending on the adaptive procedure being done - see the included 2D example.

## [Under heavy development] Adaptive grid sampler example - 2d_adaptive_example.py
Overview: This example script generates and resamples from a simple 2d exponential decay model using the grid sampler and adaptive grid sampler module. First the model f(theta) is defined and synthetic data is generated with Gaussian noise. A bounded parameter space is created from domain knowledge and is turned into a discrete randomized mesh grid determined by user set hyperparameters. This mesh grid then follows the adaptive procedure briefly described above - resulting in a final set of densely packed grid points near the high probability regions with all data included. 

-August George, Zuckerman Lab, OHSU, 2021
