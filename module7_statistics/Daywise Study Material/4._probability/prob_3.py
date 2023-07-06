from scipy.stats import geom, poisson, binom, uniform, expon

geom.pmf(3, 0.2)

geom.pmf(7, 0.1)

#######
poisson.sf(30, 25)

poisson.sf(1, 0.05)

#########
binom.sf(2, 8, 0.42)

binom.pmf(4, 8, 0.42)

############
uniform.cdf(30, loc=20, scale=40)

uniform.sf(40 ,loc=20, scale=40)

#############
expon.sf(31, loc=1/28, scale=28)
expon.cdf(25, loc=1/28, scale=28)

###############
expon.cdf(85, loc=1/85, scale=85)
expon.sf(150, loc=1/85, scale=85)
