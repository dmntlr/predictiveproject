import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix



#CSV Datei lesen, es wird beim Komma in eine neue Spalte aufgeteilt
missing_values = ["na", "", " Es verkehrt"]
data = pd.read_csv("19.06.20_travels_Frankfurt.csv",sep=',', na_values=missing_values)


#löscht Zeilen mit nan-Werten
data = data.dropna()


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


#Alle negativen Zahlen entfernen damit zu frühe abfahrtzeiten nicht gezählt werden
data = data[(data.iloc[:, 12:13] > 0).all(1)]

#Finden von Klammern und Inhalt löschen damit Modelle richtig gruppiert werden
data['TIN'] = [re.sub(r'\([^)]*\)','', str(x)) for x in data['TIN']]



#Groupiert nach Zügen und summiert dabei die Zeiten auf
data2 = data.groupby('TIN').agg({'TAc' : 'sum'}).reset_index()

#Sortiert nach Zeiten
data2 = data2.sort_values(by=['TAc'], ascending=False)

#X und Y in ganzer liste speichern
X = data2.iloc[:, 0:1].values.ravel()
y = data2.iloc[:, 1:2].values.ravel()

plt.figure(figsize=(40, 4))
plt.bar(X, y)
plt.xticks(rotation='vertical')
plt.show()

#Wahrscheinlichkeit, dass ein bestimmter Zug zu spät kommt

data1 = data.loc[:, ['TIN', 'TAc', 'TIT']]

#data1['TAc'] = [0 if x < 0 else x for x in y]

#binäre Spalte 'LATE' (1=Verspätung, 0=keine Verspätung)
late = []
cntlate = 0
cntintime = 0
for row in data1['TAc']:
      if row > 1:
         late.append(1)
         cntlate+=1
      else:
         late.append(0)
         cntintime += 1


print("Es gibt " + cntlate.__str__() + " Züge die eine Verspätung haben.")

print("Es gibt " + cntintime.__str__() + " Züge die pünktlich sind.")


data1['LATE'] = late
#nach Uhrzeiten sortiert
data1 = data1.sort_values(by=['TIT'])
#Uhrzeiten für Logistische Regression in Integer umgewandelt
#Beispiel: 11:00 Uhr = 1100; 12:30 Uhr = 1230
data1['TIT'] = data1['TIT'].str.replace(':', '').astype(int)
X = data1.iloc[:, 2:3].values
y = data1.iloc[:, 3:4].values

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=0)

X_train = scaler.inverse_transform(X_train_scaled) #transformiert die "scaled" Daten wieder zurück
X_test = scaler.inverse_transform(X_test_scaled)

log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, Y_train)
y_pred = log_reg.predict(X_test_scaled)

#plt.figure(figsize=(50, 5))
plt.scatter(X_train, Y_train, color='red', label='Training data')
plt.xlabel('Uhrzeit')
plt.ylabel('Verspätung (Ja/Nein)')
X_train = np.sort(X_train[:, 0])
X_train_sc = np.sort(X_train_scaled[:, 0])
y_pred_proba = log_reg.predict_proba(X_train_sc.reshape(-1, 1))[:, 1]
plt.plot(X_train, y_pred_proba, color='green', linewidth=3, label='Log Reg')
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='blue', label='Test data')
plt.xlabel('Uhrzeit')
plt.ylabel('Verspätung (Ja/Nein)')
plt.plot(X_train, y_pred_proba, color='green', linewidth=3, label='Log Reg')
plt.legend()
plt.show()

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix: " + str(cm))
sens = str(cm[1, 1]/(cm[1, 1] + cm[1, 0]))
print("Sensitivity: " + str(cm[1, 1]/(cm[1, 1] + cm[1, 0])))
spec = str(cm[0, 0]/(cm[0, 0] + cm[0, 1]))
print("Specificity: " + str(cm[0, 0]/(cm[0, 0] + cm[0, 1])))
print("Precision: " + str(cm[1, 1]/(cm[1, 1] + cm[0, 1])))
print("Accuracy: " + str((cm[0, 0]+cm[1, 1])/(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])))


#Vorher splitten weil ja nach datum sortiert werden muss
train, test = train_test_split(data, test_size=0.2)
#nach Uhrzeiten sortiert
train = train.sort_values(by=['TIT'])
test = test.sort_values(by=['TIT'])

X_train = train['TIT'].str.replace(':', '').astype(int)
X_train= X_train.values.reshape(-1,1)
y_train = train['TAc']

X_test = test['TIT'].str.replace(':', '').astype(int)
X_test= X_test.values.reshape(-1,1)
y_test = test['TAc']

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X_train)

poly.fit(X_poly, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)


plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, lin2.predict(X_poly), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('Uhrzeit')
plt.ylabel('Verspätung')

plt.show()

#Vorhersage für Test daten

for uhrzeit, erg in zip(np.nditer(X_test), np.nditer(y_test)):
    print("Versuche für Uhrzeit ",uhrzeit, "vorherzugagen->",lin2.predict(poly.fit_transform([[uhrzeit]])), "eigentliches Ergebniss ist" , erg)
