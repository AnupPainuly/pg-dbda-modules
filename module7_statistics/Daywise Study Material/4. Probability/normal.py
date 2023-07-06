from scipy.stats import norm
import numpy as np
norm.cdf(58, 64, 4)
norm.sf(200, 180, 30)

#############
norm.cdf(33, 34, 1.9)

norm.cdf(38, 34, 1.9) - norm.cdf(31, 34, 1.9)

norm.sf(36, 34, 1.9)

norm.sf(33, 34, 1.9)

norm.ppf(0.94, 34, 1.9)
######################
norm.cdf(600, 610, 20)

norm.cdf(620, 610, 20) - norm.cdf(590, 610, 20)

norm.sf(650, 610, 20)

norm.ppf(0.95,610, 20 )
norm.cdf(642.8970725390294, 610, 20)

values = np.array([540, 600, 650, 700])
print("Standardized Values:", (values-610)/20)

#########################
norm.sf(50, 38, 5)

norm.cdf(10, 38, 5)

norm.cdf(60, 38, 5) - norm.cdf(30, 38, 5)

##########################
norm.ppf(0.98, 313, 57)

norm.ppf(0.9, 93, 22)

pa = norm.sf(450,313,57)
pb = norm.sf(150,93,22)
pa + pb - pa*pb
