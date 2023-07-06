import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

cars93 = pd.read_csv("Cars93.csv")

cts = cars93['AirBags'].value_counts()
plt.bar(cts.index,cts)
plt.show()

plt.pie(cts,labels=cts.index)
plt.show()

plt.scatter(cars93['MPG.city'], cars93["Price"])
plt.xlabel("MPG City")
plt.ylabel("Price")
plt.show()

usa = cars93[cars93['Origin']=="USA"]
non_usa = cars93[cars93['Origin']=="non-USA"]

plt.scatter(usa['MPG.city'], usa["Price"],label="USA")
plt.scatter(non_usa['MPG.city'], 
            non_usa["Price"], label="Non-USA")
plt.xlabel("MPG City")
plt.ylabel("Price")
plt.legend()
plt.show()

sns.scatterplot(x='MPG.city',y='Price',
                hue="Origin",
                data=cars93)
plt.show()

sns.scatterplot(x='MPG.city',y='Price',
                hue="Type",
                data=cars93)
plt.show()

cars93['Price'].mean()

cts = cars93.groupby('AirBags')['Price'].mean()
plt.bar(cts.index,cts)
plt.show()

