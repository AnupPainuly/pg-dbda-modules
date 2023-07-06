from scipy.stats import poisson
import numpy as np
#######
poisson.pmf(5,12)
poisson.cdf(12,12)
poisson.sf(14,12)

np.sum(poisson.pmf(np.arange(10,16), 12))
# OR
poisson.cdf(15,12) - poisson.cdf(9,12) 

##########
poisson.sf(70, 56)
poisson.cdf(19, 56)

##########
poisson.sf(5, 4)
poisson.cdf(2, 4)
