import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

dataset = pd.read_csv("19.06.20_travels_Frankfurt.csv")

dataset = dataset.loc[:, ['TIN', 'TIR', 'TIM', 'TIRE', 'TIT', 'TAc']]

#Analyse der verspäteten Züge für die ersten 100 Zeilen
#löscht Zeilen mit nan-Werten
dataset = dataset.dropna()

X = dataset.iloc[0:100, 0:1].values
y = dataset.iloc[0:100, 5:6].values


xx = X.ravel()
yy = y.ravel()

plt.bar(xx, yy, color='red')
plt.xticks(rotation='vertical')
#plt.savefig("test.png")
plt.show()

