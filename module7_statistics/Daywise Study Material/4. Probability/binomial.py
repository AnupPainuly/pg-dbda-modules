from scipy.stats import binom

binom.pmf(5, 40, 0.25)

binom.cdf(10, 40, 0.25)

binom.sf(19, 40, 0.25)
#or
1 - binom.cdf(19, 40, 0.25)

binom.stats(40, 0.25)

##############
binom.sf(4, 50, 0.07)
#############
binom.sf(69,71,0.94)
###############
binom.pmf(0, 20, 0.35)
binom.pmf(10, 20, 0.35)
binom.cdf(10, 20, 0.35)
binom.sf(13, 20, 0.35)
