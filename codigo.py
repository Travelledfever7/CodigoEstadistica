# =========================================================
#  Modelo Híbrido Final: Regresión Lineal + Random Forest
#  Análisis Saber Pro 2018–2022
# =========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Cargar los datos
# =========================================================
data = pd.read_csv(
    r"resultados.csv",
    encoding='latin-1',
    sep=',',
    low_memory=False
)
print("✅ Archivo cargado correctamente")
print(f"🔹 Dimensiones iniciales: {data.shape[0]} filas y {data.shape[1]} columnas\n")


# =========================================================
# Variables socioeconómicas seleccionadas
# =========================================================
features = [
    # 🏠 Socioeconómicas
    'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
    'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA',

    # 🎓 Académicas
    'ESTU_NIVEL_PRGM_ACADEMICO', 'ESTU_METODO_PRGM',
    'ESTU_HORASSEMANATRABAJA', 'ESTU_ESTADOINVESTIGACION',

    # 💰 Financieras
    'ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO',
    'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO',

    # 🏫 Institucionales / Geográficas
    'INST_CARACTER_ACADEMICO', 'INST_ORIGEN',
    'ESTU_DEPTO_PRESENTACION', 'ESTU_MCPIO_PRESENTACION',

    # 👩‍🎓 Demográficas
    'ESTU_GENERO', 'ESTU_NACIONALIDAD',
    'ESTU_PRIVADO_LIBERTAD', 'ESTU_NUCLEO_PREGRADO',

    # 🌎 Contextuales
    'PERIODO', 'MOD_RAZONA_CUANTITAT_PUNT'
]

# =========================================================
# Variable objetivo
# =========================================================
data['PUNTAJE_PROMEDIO'] = data[[
    'MOD_RAZONA_CUANTITAT_PUNT',
    'MOD_LECTURA_CRITICA_PUNT',
    'MOD_COMUNI_ESCRITA_PUNT',
    'MOD_INGLES_PUNT',
    'MOD_COMPETEN_CIUDADA_PUNT'
]].mean(axis=1)
target = 'PUNTAJE_PROMEDIO'

# =========================================================
# LIMPIEZA DE DATOS
# =========================================================
print(" Iniciando limpieza de datos...")

data = data.drop_duplicates()

# Normalizar texto
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype(str).str.strip().str.upper()

# Eliminar columnas con más del 90 % de nulos
cols_to_drop = data.columns[data.isnull().mean() > 0.9].tolist()
data = data.drop(columns=cols_to_drop)
print(f"🗑️ Columnas eliminadas (>90 % nulos): {cols_to_drop}")

# Rellenar nulos
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna('DESCONOCIDO')
    else:
        data[col] = data[col].fillna(0)

print(f" Dimensiones después de limpieza: {data.shape}\n")

# =========================================================
# Codificación de variables categóricas
# =========================================================
le = LabelEncoder()
for col in features:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# =========================================================
# División de datos
# =========================================================
data = data.sample(frac=0.3, random_state=42)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# MODELO HÍBRIDO: Regresión Lineal + Random Forest
# =========================================================
# Entrenar la parte lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lineal = lr.predict(X_train)

# Calcular residuos
residuos_train = y_train - pred_lineal

# Entrenar el Random Forest para los residuos
rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, residuos_train)

# Predicción final combinada
pred_lineal_test = lr.predict(X_test)
correccion_rf = rf.predict(X_test)
y_pred_final = pred_lineal_test + correccion_rf

# =========================================================
# Evaluación final del modelo híbrido
# =========================================================
r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
precision = r2 * 100  # porcentaje de predicción

print("📈 Resultados del Modelo Híbrido Final:")
print(f"Coeficiente de determinación (R²): {r2:.3f}")
print(f"Error absoluto medio (MAE): {mae:.3f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.3f}")
print(f" Porcentaje de predicción del modelo: {precision:.2f}%\n")

# =========================================================
# Análisis de Coeficientes e Importancia
# =========================================================
coef_df = pd.DataFrame({
    'Variable': features,
    'Coeficiente (Lineal)': lr.coef_
}).sort_values(by='Coeficiente (Lineal)', ascending=False)

importancias = pd.DataFrame({
    'Variable': features,
    'Importancia (No lineal)': rf.feature_importances_
}).sort_values(by='Importancia (No lineal)', ascending=False)

print("Coeficientes (parte lineal):\n", coef_df)
print("\nImportancia (parte no lineal):\n", importancias)

# =========================================================
# MATRIZ DE CORRELACIÓN
# =========================================================
corr = data[features + [target]].corr()
plt.figure("Matriz de correlación", figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación - Variables socioeconómicas y puntaje Saber Pro")
plt.tight_layout()

# =========================================================
# GRÁFICA REAL VS PREDICHO
# =========================================================
plt.figure("Predicción vs Real", figsize=(7, 7))
plt.scatter(y_test, y_pred_final, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Comparación entre valores reales y predichos (Modelo Híbrido)")
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.grid(True)
plt.tight_layout()

# =========================================================
# GRÁFICAS ADICIONALES
# =========================================================
plt.figure("Coeficientes Lineales", figsize=(8, 5))
plt.barh(coef_df['Variable'], coef_df['Coeficiente (Lineal)'])
plt.title("Coeficientes - Parte Lineal del Modelo")
plt.xlabel("Coeficiente")
plt.tight_layout()

plt.figure("Importancia No Lineal", figsize=(8, 5))
plt.barh(importancias['Variable'], importancias['Importancia (No lineal)'])
plt.title("Importancia - Parte No Lineal (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()

plt.figure("Distribución Puntaje", figsize=(8, 5))
plt.hist(data['PUNTAJE_PROMEDIO'], bins=30)
plt.title("Distribución del Puntaje Promedio Saber Pro")
plt.xlabel("Puntaje promedio")
plt.ylabel("Número de estudiantes")
plt.tight_layout()

plt.show()
