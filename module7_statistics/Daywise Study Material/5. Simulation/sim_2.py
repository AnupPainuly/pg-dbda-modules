import numpy as np
import pandas as pd

# Generates random number for uniform 
# distribution (0,1)
np.random.seed(87)
samp = np.random.rand(4)
print(samp)


samp = np.random.rand(7)
np.where(samp<0.5, "H", "T")

##### Sick Drivers
x = np.array([0,1,2,3,4,5])
p = np.array([0.3,0.2,0.15,0.1,0.13,0.12])
cp = np.cumsum(p)
print(cp)
e_x = np.sum(x*p)
print("E(X) =",e_x)
np.random.seed(87)
samp = np.random.rand(30)
print(samp)

r = 0.24074492
ind_sim = np.sum(cp < r)
x[ind_sim]

sim_data = []
for r in samp:
    ind_sim = np.sum(cp < r)
    sim_data.append(x[ind_sim])

reserved = 2
sim_data = np.array(sim_data)

days = np.sum(sim_data > reserved)
print("Days buses did not run =", days)

buses = np.where(sim_data - reserved<0, 0,
                 sim_data - reserved)
drivers = pd.DataFrame({'Simulated':sim_data,
                        'Buses_Cancelled':buses})
print(drivers)
print("Buses did not run =", np.sum(buses))

############ Supply - Demand
supp = np.array([10,20,30,40,50])
supp_days = np.array([40,50,190,150,70])
supp_prob = supp_days/500
supp_cp = np.cumsum(supp_prob)

np.random.seed(87)
samp = np.random.rand(30)

sim_supp = []
for r in samp:
    ind_sim = np.sum(supp_cp < r)
    sim_supp.append(supp[ind_sim])

dem = np.array([10,20,30,40,50])
dem_days = np.array([50,110,200,100,40])
dem_prob = dem_days/500
dem_cp = np.cumsum(dem_prob)

np.random.seed(78)
samp = np.random.rand(30)

sim_dem = []
for r in samp:
    ind_sim = np.sum(dem_cp < r)
    sim_dem.append(dem[ind_sim])

sales_df = pd.DataFrame({'Sim_Supply':sim_supp,
                         'Sim_Demand':sim_dem})

sales_df['Sold'] = np.minimum(sales_df['Sim_Supply'],
                              sales_df['Sim_Demand'])
sales_df['Perished'] = np.where(sales_df['Sim_Supply']>sales_df['Sim_Demand'],
                                sales_df['Sim_Supply']-sales_df['Sim_Demand'],
                                0)
sales_df['Profit'] = sales_df['Sold']*10 
sales_df['Loss'] = sales_df['Perished']*8
sales_df['Net Profit'] = sales_df['Profit'] - sales_df['Loss']
