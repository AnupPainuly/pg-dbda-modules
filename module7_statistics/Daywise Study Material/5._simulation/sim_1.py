#### Simulation
from scipy.stats import binom,poisson,norm, expon

## Insurance agent for 24 months
sim_ins = binom.rvs(n=40, p=0.25, size=24,
                    random_state=87)
sim_ins.mean()

## Disorder for every 20 patients, 10 times
sim_ins = binom.rvs(n=20, p=0.35, size=10,
                    random_state=87)
sim_ins.mean()

## Customers call per day for 30 days
sim_calls = poisson.rvs(mu=56, size=30,
                    random_state=87) 
sim_calls.mean()

## Bike Mileages for 100 bikes
sim_bikes = norm.rvs(loc=34, scale=1.9, size=100,
                     random_state=87)
sim_bikes.mean()

## pistons for 200 batches
sim_pistons = binom.rvs(n=10, p=0.12, size=200,
                        random_state=87)
sim_pistons.mean()
print("Minimum rejected =", sim_pistons.min())
print("Maximum rejected =", sim_pistons.max())

## life of 200 batteries
sim_bat = expon.rvs(loc=1/4, scale=4, size=200,
                    random_state=87)
sim_bat.mean()
