import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
#CSV Datei lesen, es wird beim Komma in eine neue Spalte aufgeteilt
data=pd.read_csv("19.06.20_travels_Frankfurt.csv",sep=',')

#Zugriff auf jede Spalte
TAA = data["TAA"]
TA = data["TA"]
TIN = data["TIN"]
TIR = data["TIR"]
TSI = data["TSI"]
TIM = data["TIM"]
TIL = data["TIL"]
TIRE = data["TIRE"]
TIP = data["TIP"]
TIT = data["TIT"]
TID = data["TID"]
TSC = data["TSC"]
TAC = data["TAc"]

#Ausgabe jeder Zeile für eine Spalte
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
      print(TIN)



#löscht Zeilen mit nan-Werten
dataset = data.dropna()
#Alle negativen Zahlen entfernen damit zu frühe abfahrtzeiten nicht gezählt werden
dataset = dataset[(dataset.iloc[:,12:13] > 0).all(1)]
#Groupiert nach Zügen und summiert dabei die Zeiten auf
dataset = dataset.groupby('TIN').agg({'TAc' : 'sum'}).reset_index()
#Sortiert nach Zeiten
dataset = dataset.sort_values(by=['TAc'], ascending=False)


X = dataset.iloc[:, 0:1].values.ravel()
y = dataset.iloc[:, 1:2].values.ravel()

plt.figure(figsize=(40, 4))
plt.bar(X,y)
plt.xticks(rotation='vertical')
plt.show()
