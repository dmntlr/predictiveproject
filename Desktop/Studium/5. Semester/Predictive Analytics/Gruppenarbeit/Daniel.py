import pandas as pd
import matplotlib.pyplot as plt

missing_values = ["na","--"]
# read csv into a dataframe
propdata=pd.read_csv("19.06.20_travels_Frankfurt.csv", sep=',' ,  na_values= missing_values)

print(propdata)
