import numpy as np
import pandas as pd
import math

class ParameterVector:

    def __init__(self, p_name=None, p_min=None, p_max=None, p_div=None):
        self.name = p_name
        self.min = p_min
        self.max = p_max
        self.div = p_div
        self.vector = np.linspace(self.min, self.max, self.div)
        self.spacing = np.abs(self.vector[1]-self.vector[0])


    def randomize(self, seed):
        np.random.seed(seed)
        rand_n = np.random.uniform(-self.spacing, self.spacing)
        self.vector = self.vector + rand_n
        self.min = np.min(self.vector)
        self.max = np.max(self.vector)


class Grid:

    def __init__(self, p_list):
        self.p_list = p_list
        self.p_vectors = [p.vector for p in p_list]
        self.p_names = [p.name for p in p_list]


    def update(self, p_list):
        self.p_list = p_list
        self.p_vectors = [p.vector for p in p_list]
        self.p_names = [p.name for p in p_list]


    def mesh_grid(self, p_vectors):
        self.mg = np.meshgrid(*self.p_vectors)
        self.mg_df = pd.DataFrame(np.vstack(list(map(np.ravel, self.mg))).T, columns=self.p_names)
        return self.mg, self.mg_df


    def randomize_grid(self, seed=1234):
        p_list_rand = []
        # for each vector object in list
        for pv_i in self.p_list:
            pv_i.randomize(seed)
            p_list_rand.append(pv_i)
        self.update(p_list_rand)
        return self.mesh_grid(self.p_vectors)



def calc_logl(x,mu,sigma):
    """ calculates the Normal log-likelihood probability:
        p(x|mu,sigma^2) = -n/2 ln(2*pi*sigma^2) - 1/2*sigma^2 sum(x-mu)^2
        where x: y_pred, mu: y_obs, sigma: error in observations, and n: # of points
    """
    n = len(mu)  # y_obs
    logp = -n*math.log(2*math.pi*(sigma**2))/2 - np.sum(((x-mu)**2)/(2 * (sigma**2)))
    return logp


def calc_rel_logp(logp):
    """ relative log probability (log likelihood): ln(p) --> ln(p/pmax) = ln(p) - ln(p_max)"""
    return logp - np.max(logp)


def calc_rel_p(rel_logp):
    """ relative probability (likelihood): p/pmax =  e**ln(p/pmax)"""
    return np.exp(rel_logp)


def calc_norm_weights(p):
    """ weight for resampling: sum(p/pmax) = 1"""
    return p / np.sum(p)


def calc_ess(rel_p):
    """ effective sample size (ESS): sum(p/pmax) = sum(relative p)"""
    return np.sum(rel_p)


def resample(df, n, w):
    """ resample n points from df using normalized weights w"""
    return df.sample(n=n, replace=True, weights=w)


def calc_log_batch_p(ess, logp):
    """ probability within a batch --> batch weight: sum(p) --> ln(sum(p)) = ln(ESS) + ln(pmax)"""
    return np.log(ess) + np.max(logp)


def calc_adjusted_ess( ess, logp, logp_max_b):
    y = np.exp(np.log(ess) + np.max(logp) - logp_max_b)
    return y





if __name__ == '__main__':
    pass