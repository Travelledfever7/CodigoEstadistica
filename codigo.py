
#  Análisis Saber Pro 2018–2022 - Random Forest Regressor


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\PC\Downloads\Estadistica\Estadistica\resultados.csv", encoding='latin-1', sep=',', low_memory=False)
print("Archivo cargado correctamente")


features = [
    'FAMI_ESTRATOVIVIENDA',
    'FAMI_EDUCACIONPADRE',
    'FAMI_EDUCACIONMADRE',
    'FAMI_TIENECOMPUTADOR',
    'FAMI_TIENEINTERNET',
    'FAMI_TIENEAUTOMOVIL',
    'FAMI_TIENELAVADORA',
    'ESTU_HORASSEMANATRABAJA'
]

# Crear la variable objetivo: puntaje promedio
data['PUNTAJE_PROMEDIO'] = data[[
    'MOD_RAZONA_CUANTITAT_PUNT',
    'MOD_LECTURA_CRITICA_PUNT',
    'MOD_COMUNI_ESCRITA_PUNT',
    'MOD_INGLES_PUNT',
    'MOD_COMPETEN_CIUDADA_PUNT'
]].mean(axis=1)

target = 'PUNTAJE_PROMEDIO'


data = data.dropna(subset=features + [target])

# Codificar variables categóricas
le = LabelEncoder()
for col in features:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])


X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(
    n_estimators=200,       # número de árboles
    max_depth=None,         # profundidad automática
    random_state=42,
    n_jobs=-1               # usar todos los núcleos del CPU
)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Coeficiente de determinación (R²): {r2:.3f}")
print(f"Error absoluto medio (MAE): {mae:.3f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.3f}\n")

importancias = pd.DataFrame({
    'Variable': features,
    'Importancia': rf.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("Importancia de las variables en el modelo Random Forest:\n")
print(importancias)
# Importancia de variables
plt.figure("Importancia de variables", figsize=(8, 5))
importancias = importancias.sort_values(by='Importancia', ascending=True)
plt.barh(importancias['Variable'], importancias['Importancia'], color='skyblue')
plt.xlabel("Importancia")
plt.title("Importancia de las variables socioeconómicas en el puntaje Saber Pro")
plt.tight_layout()

# Distribución del puntaje promedio
plt.figure("Distribución del puntaje", figsize=(8,5))
plt.hist(data['PUNTAJE_PROMEDIO'], bins=30, color='teal', edgecolor='black')
plt.title("Distribución del puntaje promedio Saber Pro")
plt.xlabel("Puntaje promedio")
plt.ylabel("Número de estudiantes")
plt.tight_layout()

# Mostrar todas las gráficas al final
plt.show()
