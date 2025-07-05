from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

X = np.array([
    [75, 98], [80, 99], [60, 97],
    [45, 95], [120, 88], [130, 85]
])
y = np.array([0, 0, 0, 1, 1, 1])

modelo = RandomForestClassifier()
modelo.fit(X, y)

joblib.dump(modelo, 'modelo_oximetro.pkl')
print("Modelo guardado como modelo_oximetro.pkl")