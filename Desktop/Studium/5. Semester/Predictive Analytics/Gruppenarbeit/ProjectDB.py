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
      print(TAC)



#Analyse der verspäteten Züge für die ersten 100 Zeilen
#löscht Zeilen mit nan-Werten
dataset = data.dropna()
#Sortieren nach verspätung
dataset = dataset.sort_values(by=['TAc'], ascending=False)

X = dataset.iloc[:, 2:3].values.ravel()
y = dataset.iloc[:, 12:13].values.ravel()


#Daten bereinigen da auch hohe negative Zahlen als Fehlzeiten drinne sind Da für TAC ja TA - TIT gerechnet wird
y = [0 if x<0 else x for x in y]

plt.figure(figsize=(40, 4))
plt.bar(X,y)
plt.xticks(rotation='vertical')

plt.show()
