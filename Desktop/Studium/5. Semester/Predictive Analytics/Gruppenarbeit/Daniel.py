import pandas as pd
import matplotlib.pyplot as plt

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
    print(TAA)