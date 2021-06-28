import numpy as np
import pandas as pd


class ParameterVector:
    """
    Convienience class to make parameter vectors for grid search
    """

    def __init__(self, p_name=None, p_min=None, p_max=None, p_div=None, pad=None):
        """
        Generates linear spacing within vector space and stores vector spacing
        :param p_name: name of parameter - e.g. k_on
        :param p_min: minimum value for parameter vector space
        :param p_max: maximum value for parameter vector space
        :param p_div: number of parameter points in parameter vector space
        :param pad: a number that sets the boundary layer allowing for shifting during randomization
        note: the vector is a linear spacing between min+pad and max-pad.
        """
        self.name = p_name
        self.b_min = p_min  # global boundary min
        self.b_max = p_max  # global boundary max
        self.min = p_min  # current min of the vector - can change during update/randomization
        self.max = p_max  # current min of the vector - can change during update/randomization
        self.div = p_div
        self.pad = pad
        self.vector, self.spacing = np.linspace(self.b_min+self.pad, self.b_max-self.pad, self.div, retstep=True)
        self.rand_n = 0  # random number to shift by (if randomization is used)


    def randomize(self, seed):
        """
        Shift each value by the same uniform random number within the spacing.
        Note: values can be shifted +/- 1/2 of the spacing
        Note 2: it's possible that the shifted values could be outside the boundary!
        :param seed: sets the randomization seed (numpy)
        :return:
        """
        np.random.seed(seed)
        rand_n = np.random.uniform(-self.spacing/2, self.spacing/2)
        rand_vector  = self.vector + rand_n

        if np.min(rand_vector) < self.b_min or np.max(rand_vector) > self.b_max:  # check if random vector is in bounds
            print(f'Randomized number is out of bounds! {np.min(rand_vector)} < {self.b_min} or {np.max(rand_vector)} > {self.b_max} for {self.name}')
            raise ValueError(f'Randomized number is out of bounds! {np.min(rand_vector)} < {self.b_min} or {np.max(rand_vector)} > {self.b_max} for {self.name}')
        else:
            self.vector = rand_vector
            self.min = np.min(self.vector)
            self.max = np.max(self.vector)
            self.rand_n = rand_n


    def change_density(self, n_points, pad=None):
        """
        Update linear spaced vector based on inputted number of points
        :param n_points: number of points to place in vector
        :return:
        """
        self.div = n_points
        # note: spacing is constant, so it is the same before and after adding the 'randomization'
        temp_vector, self.spacing = np.linspace(self.b_min+self.pad, self.b_max-self.pad, self.div, retstep=True)
        self.vector = temp_vector + self.rand_n


class MeshGrid:
    """
    General grid class built from parameter vectors. Has methods to update, create a mesh grid, and randomize mesh grid.
    """

    def __init__(self, p_list):
        """
        Initializes and stores parameter information for Grid object
        :param p_list: list of Parameter Vector objects (custom class)
        """
        self.p_list = p_list
        self.p_vectors = [p.vector for p in p_list]
        self.p_names = [p.name for p in p_list]


    def update(self, p_list):
        """
        Updates the parameter vector information for the Grid object
        :param p_list: list of Parameter Vector objects (custom class)
        :return:
        """
        self.p_list = p_list
        self.p_vectors = [p.vector for p in p_list]
        self.p_names = [p.name for p in p_list]


    def mesh_grid(self, p_vectors):
        """
        Creates a mesh grid from a list of parameter vectors (NOT the custom Parameter Vector class)
        :param p_vectors: a list of parameter vectors - e.g. [[0,1,2], [3,4,5]]
        :return: a numpy meshgrid, and a pandas dataframe of the meshgrid
        """
        self.mg = np.meshgrid(*self.p_vectors)
        self.mg_df = pd.DataFrame(np.vstack(list(map(np.ravel, self.mg))).T, columns=self.p_names)
        return self.mg, self.mg_df


    def randomize_grid(self, seed=1234):
        """
        Randomize the grid coordinates while maintaining the spacing. Calls the Parameter Vector randomization method.
        :param seed: randomization seed
        :return: returns results of calling the mesh_grid func. - a numpy meshgrid and a pandas dataframe
        """
        p_list_rand = []
        for pv_i in self.p_list:
            pv_i.randomize(seed)
            p_list_rand.append(pv_i)
        self.update(p_list_rand)
        return self.mesh_grid(self.p_vectors)



def calc_logl(x,mu,sigma):
    """ calculates the Normal log-likelihood probability:
        p(x|mu,sigma^2) = -n/2 ln(2*pi*sigma^2) - 1/2*sigma^2 sum(x-mu)^2
        where x: y_pred, mu: y_obs, sigma: error in observations, and n: # of observed points
    """
    n = len(mu)
    logp = -n*np.log(2*np.pi*(sigma**2))/2 - np.sum(((np.array(x)-np.array(mu))**2)/(2 * (sigma**2)))
    return logp


def calc_rel_logp(logp):
    """ relative log probability (log-likelihood):
    log-likelihood = log(p) --> log(p/pmax) = log(p) - ln(p_max)
    """
    return logp - np.max(logp)


def calc_rel_p(rel_logp):
    """ relative probability (likelihood):
    rel_p = p/pmax = e**log(p/pmax)
    """
    return np.exp(rel_logp)


def calc_norm_weights(p):
    """ weight used for resampling (so the sum of pobability = 1):
    sum(p) = 1 -> w = p/sum(p)
    """
    return p / np.sum(p)


def calc_ess(rel_p):
    """ effective sample size (ESS):
    ESS = sum(p/pmax) = sum(relative p)
    """
    return np.sum(rel_p)


def resample(df, n, w):
    """ resample n points from df using normalized weights w"""
    return df.sample(n=n, replace=True, weights=w)


def calc_log_batch_p(ess, logp):
    """ probability within a batch --> batch weight:
    sum(p) --> log(sum(p)) = log(ESS) + log(pmax)"""
    return np.log(ess) + np.max(logp)


def calc_adjusted_ess( ess, logp, logp_max_b):
    """ ess adjusted for batches (needs testing):
    adjusted ESS = (p_max/p_max_all_batches)*ESS = (p_max/p_max_all_batches) * sum(p/p_max)
    --> log adjusted ESS = log(p_max) - log(p_max_all_batches) + log(ESS)
    """
    return np.exp(np.log(ess) + np.max(logp) - logp_max_b)



if __name__ == '__main__':
    pass