import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
img = Image.open(r"C:\Training\Academy\Deep Learning Material\Images\Simple Images\Shapes\1.jpg")
maskImg = np.array(img) 

maskImg = maskImg.mean(axis=2)



plt.imshow(maskImg[:,:,0],cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(maskImg[:,:,1],cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(maskImg[:,:,2],cmap="gray")
plt.axis("off")
plt.show()

########## Bird 
img = Image.open(r"C:\Training\Academy\Deep Learning Material\Images\bird1.png")
maskImg = np.array(img) 

maskImg = maskImg.mean(axis=2)

plt.imshow(maskImg,cmap="gray")
plt.axis("off")
plt.show()
