import numpy as np 
import pandas as pd 

x = np.array([23, 45, 89])
p = np.array([0.625, 0.1734694, 0.2015306])

print("E(X) =", np.sum(x*p))

x = np.array([1,2,3,4])
f = np.array([15,31,24,7])
p = f/np.sum(f)

e_x = np.sum(x*p)
print("E(X) =", e_x)
e_x_2 = np.sum((x**2)*p)
v_x = e_x_2 - (e_x**2)
print("Var(X) =", v_x)
print("Std(X) =", np.sqrt(v_x))


################# Joint ########################
import numpy as np
a = np.array([[125,	18,	17,	50,	8],
              [103,	8,	10,	92,	4]])
total = a.sum()
joint = a/total
print("Joint Distribution =\n",joint)

m_c = np.sum(joint, axis=0)
print("Column Marginal =\n", m_c)
m_r = np.sum(joint, axis=1)
print("Row Marginal =\n",m_r)

cond_c = joint/m_c
print("Conditional Distribution by Columns =\n", cond_c)

cond_r1 = joint[0,:]/m_r[0]
cond_r2 = joint[1,:]/m_r[1]
print("Conditional Distribution by Rows =\n", cond_r1, cond_r2)

##### Prob 2
a = np.array([[56,42],
              [43,42],
              [62,37],
              [100,90]])
total = a.sum()
joint = a/total
print("Joint Distribution =\n",joint)
joint_df = pd.DataFrame(joint, columns=['Book','DVD'],
                        index=['East','North','South','West'])
m_c = np.sum(joint, axis=0)
print("Column Marginal =\n", m_c)
m_r = np.sum(joint, axis=1)
print("Row Marginal =\n",m_r)

cond_c = joint/m_c
print("Conditional Distribution by Columns =\n", cond_c)

cond_r1 = joint[0,:]/m_r[0]
cond_r2 = joint[1,:]/m_r[1]
cond_r3 = joint[2,:]/m_r[2]
cond_r4 = joint[3,:]/m_r[3]
print("Conditional Distribution by Rows =")
print(cond_r1,cond_r2,cond_r3,cond_r4)

####### Cars93
cars93 = pd.read_csv("Cars93.csv")
## Joint & Marginal
pd.crosstab(index=cars93['Type'],
            columns=cars93['AirBags'] ,
            margins=True, normalize=True)




