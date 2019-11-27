import pandas as pd
import matplotlib.pyplot as plt

missing_values = ["na","--"]
# read csv into a dataframe
propdata=pd.read_csv("data.csv", na_values= missing_values)

plt.plot(propdata)
plt.show()