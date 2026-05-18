import time
import numpy as np

#distribution:
# UNIFORM
# GAUSSIAN
# SAMPLE
# BINQMIAL
# UNIQUE

random_data_str = '''{{
    seed: {}
    upper_bound: {}
    lower_bound: {}
    nan_dropout: {}
    inf_dropout: {}
    zero_dropout: {}
    positive: {}
    negative: {}
    distribution: {}
  }}
'''

gaussian_data_str = '''{{
    seed: {}
    mu: {}
    sigma: {}
    nan_dropout: {}
    inf_dropout: {}
    zero_dropout: {}
    positive: {}
    negative: {}
    distribution: {}
  }}
'''


class RandomData():
    tmp = int(time.time())
    np.random.seed(tmp)
    seed = 0

    # unifom and unique
    lower_bound = -100
    upper_bound = 100
    lower_bound_float = -100
    lower_bound_float = -100

    # guassian
    mu = 1
    sigma = 1
    mu_double = 1
    sigma_double = 1
    distribution = 'UNIFORM'
    convert_dtype = 'false'

    # dropout
    zero_dropout = 0.0
    nan_dropout = 0.0
    inf_dropout = 0.0
    positive = 'false'
    negative = 'false'

    def __init__(self):
        self.seed = np.random.randint(self.tmp)
        pass

    def str(self):
        if self.distribution == 'GAUSSIAN':
            return gaussian_data_str.format(self.seed, self.mu, self.sigma,
                                            self.nan_dropout, self.inf_dropout,
                                            self.zero_dropout, self.positive,
                                            self.negative, self.distribution)
        else:
            return random_data_str.format(self.seed, self.upper_bound,
                                          self.lower_bound, self.nan_dropout,
                                          self.inf_dropout,
                                          self.zero_dropout, self.positive,
                                          self.negative, self.distribution)
