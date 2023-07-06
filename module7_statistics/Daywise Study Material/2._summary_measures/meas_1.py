import numpy as np 

a = np.array([20,89,0,29,102,58,8,79])
b = np.array([43,28,28,29,37,38,28,33,35])

print("Mean =",np.mean(a))
rng = np.max(a) - np.min(a)
print("Range =", rng)
q1 = np.quantile(a, 0.25)
q3 = np.quantile(a, 0.75)
qd = (q3-q1)/2
print("Quartile Deviation =", qd)

mad_a = np.mean(np.absolute(a - np.mean(a)))
print("Mean Deviation =", mad_a)

var_a = np.mean((a - np.mean(a))**2)
print("Variance =", var_a)
np.var(a)

print("Mean =", np.mean(b))

rng = np.max(b) - np.min(b)
print("Range =", rng)
q1 = np.quantile(b, 0.25)
q3 = np.quantile(b, 0.75)
qd = (q3-q1)/2
print("Quartile Deviation =", qd)

mad_b = np.mean(np.absolute(b - np.mean(b)))
print("Mean Deviation =", mad_b)

var_b = np.mean((b - np.mean(b))**2)
print("Variance =", var_b)
np.var(b)

#####################################################

def stats(x):
    print("Mean =", np.mean(x))
    print("Variance =", np.var(x))
    print("Std Deviation =", np.std(x))
    print("Coeff of Variation =", (np.std(x)/np.mean(x))*100)
    
weight = np.array([65,78,45,90,73,40])
stats(weight)  

height = np.array([169,170,132,100,137,155])
stats(height)
