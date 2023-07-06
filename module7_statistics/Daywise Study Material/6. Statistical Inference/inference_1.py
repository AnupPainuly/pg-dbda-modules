from scipy.stats import norm
from scipy.stats import binom
import matplotlib.pyplot as plt 

## Bike Mileages for 100 bikes
sim_bikes1 = norm.rvs(loc=34, scale=1.9, size=100)
sim_bikes1.mean()

sim_bikes2 = norm.rvs(loc=34, scale=1.9, size=100)
sim_bikes2.mean()


sim_bikes = norm.rvs(loc=34, scale=1.9, size=1000)
sim_bikes.mean()


## Insurance agent 
sim_ins = binom.rvs(n=40, p=0.25, size=5000,
                    random_state=87)
sim_ins.mean()
# theoretical mean E(X) = np = 10

## Sampling Dist

means = []
for i in range(1, 101):
    sim_bikes1 = norm.rvs(loc=34, scale=1.9, size=50)
    means.append(sim_bikes1.mean())

print(len(means))

plt.hist(means)
plt.title("Sampling Distribution")
plt.show()



## Insurance agent 
sim_ins = binom.rvs(n=40, p=0.25, size=40000,
                    random_state=87)
sim_ins.mean()
# theoretical mean E(X) = np = 10


means = []
for i in range(1, 101):
    sim_ins = binom.rvs(n=40, p=0.25, size=40000)
    means.append(sim_ins.mean())

print(len(means))
plt.hist(means)
plt.title("Sampling Distribution")
plt.show()





