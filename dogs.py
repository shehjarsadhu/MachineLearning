##Predicting a dog species
import numpy as np
import matplotlib.pyplot as plt

greyhound = 500
labs = 500

gery_height = 28 + 4 * np.random.randn(greyhound)
lab_height = 24 + 4 * np.random.randn(labs)
plt.hist([gery_height,lab_height], stacked = True ,color =['r','b'])
plt.show()
