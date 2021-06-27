# Development
Notes and to-do items for the projects in this repository.
Updated: June 27, 2021.

## Grid sampler module - grid.py
Mostly stable but needs unit tests and batching to be verified
* known issues
    * Adjusted ESS and batch probability aren't stable
    * Slow / memory problems for large dataframes
    * Issues w/ configuring HPC and ray
* unit tests 
    * Parameter Vector should always stay in boundary
    * reality checks for helper functions
    * sigma > 0
    * **verify batch probability and adjusted ESS calculations are working!**
* refactor
    * improve doc-strings  
    * Parameter Vector min/max b_min/b_max
    * handling out of bounds Parameter Vector values
    * create mesh grid --> randomize mesh grid redundancy

## Grid sampler example - single_batch_example.py
Should add example for multiple batches
* known issues: 
    * setting the data / filename


## [Under heavy development] Adaptive grid sampler module - agrid.py 
Not very stable, only tested on a single example and it was very slow. 
* known issues:
    * **poor performance - need to fix collision detection and try to avoid the loops in Pandas. Should do profiler check!**
    * need a boundary check during expansion (might be ok because of 'padding' that is added to each side of the parameter space)
    * during initialization the grid keeps randomly generating grid points until they are within the boundaries - should find a better way
    * collion detection - alphabetical string lists might be better than the sets
    * rounding error when using numerical IDs (1e-17 digits) - rounding to nearest 1e-15 digits
* unit tests:
    * tracking the correct grid levels and spacing at each iteration
    * ESS expansion routine: ESS should not go up during expansion with added datapoint!
* refactor: 
    * hardcoded df['w'] when sorting df weights - should change 'w' to variable
    * automate the expanison / packing list creation
    * tracking grids w/ lists but these might become too large for realistic problems
    * grid reduction method - should append to a list and sum the list rather than summing df[:i] each iteration
    * too many lists and indices are making the code logic confusing to read, need to clean up the logic and refactor the variable names
    * make seperate function for 'while ESS < ESS_min' loop which is repeated a few times - don't repeat yourself

## [Under heavy development] Adaptive grid sampler example - 2d_adaptive_example.py
Extremely slow for a toy 2d example. See above. 


*August George, Zuckerman Lab, OHSU*
