import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
#CSV Datei lesen, es wird beim Komma in eine neue Spalte aufgeteilt
data = pd.read_csv("19.06.20_travels_Frankfurt.csv",sep=',')

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
dataset = dataset[(dataset.iloc[:, 12:13] > 0).all(1)]
#Groupiert nach Zügen und summiert dabei die Zeiten auf
dataset = dataset.groupby('TIN').agg({'TAc' : 'sum'}).reset_index()
#Sortiert nach Zeiten
dataset = dataset.sort_values(by=['TAc'], ascending=False)


X = dataset.iloc[:, 0:1].values.ravel()
y = dataset.iloc[:, 1:2].values.ravel()

plt.figure(figsize=(40, 4))
plt.bar(X, y)
plt.xticks(rotation='vertical')
plt.show()

#Wahrscheinlichkeit, dass ein bestimmter Zug zu spät kommt

data1 = dataset.loc[:, ['TIN', 'TAc']]
data1['TAc'] = [0 if x < 0 else x for x in y]

#binäre Spalte 'LATE' (1=Verspätung, 0=keine Verspätung)
late = []
for row in data1['TAc']:
      if row > 0:
         late.append(1)
      else:
         late.append(0)

data1['LATE'] = late

X = data1.iloc[:, 0:1].values.ravel()
y = data1.iloc[:, 2:3].values.ravel()

plt.scatter(X, y)
plt.xticks(rotation='vertical')
plt.show()