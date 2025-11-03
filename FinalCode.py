

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

print(" Iniciando modelo híbrido (25 variables)...\n")


data = pd.read_csv(
    r"\resultados.csv",
    encoding='latin-1',
    sep=',',
    low_memory=False
)
print(f" Archivo cargado: {data.shape[0]} filas, {data.shape[1]} columnas")


features = [
   
    'FAMI_ESTRATOVIVIENDA', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
    'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA',

   
    'ESTU_NIVEL_PRGM_ACADEMICO', 'ESTU_METODO_PRGM',
    'ESTU_HORASSEMANATRABAJA', 'ESTU_ESTADOINVESTIGACION',

    
    'ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO',
    'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO',


    'INST_CARACTER_ACADEMICO', 'INST_ORIGEN',
    'ESTU_DEPTO_PRESENTACION', 'ESTU_MCPIO_PRESENTACION',

 
    'ESTU_GENERO', 'ESTU_NACIONALIDAD',
    'ESTU_PRIVADO_LIBERTAD', 'ESTU_NUCLEO_PREGRADO',


    'PERIODO', 'MOD_RAZONA_CUANTITAT_PUNT'
]


print(" Calculando puntaje promedio...")
data['PUNTAJE_PROMEDIO'] = data[[
    'MOD_RAZONA_CUANTITAT_PUNT',
    'MOD_LECTURA_CRITICA_PUNT',
    'MOD_COMUNI_ESCRITA_PUNT',
    'MOD_INGLES_PUNT',
    'MOD_COMPETEN_CIUDADA_PUNT'
]].mean(axis=1)
target = 'PUNTAJE_PROMEDIO'


print(" Limpiando datos...")
data = data.drop_duplicates()

for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype(str).str.strip().str.upper()

data = data.dropna(subset=features + [target])
data = data[(data[target] >= 50) & (data[target] <= 250)]

print(f" Datos limpios: {data.shape[0]} filas, {len(features)} columnas.\n")


print(" Codificando texto y limpiando valores...")

le = LabelEncoder()
for col in features:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

def limpiar_numeros(valor):
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, str):
        valor = re.sub(r'[\[\]\(\)\s,]', '', valor)
    try:
        return float(valor)
    except ValueError:
        return np.nan

for col in features:
    data[col] = data[col].apply(limpiar_numeros)

data = data.fillna(data.median(numeric_only=True))
print(" Conversión numérica completa.\n")


print(" Creando variables derivadas...")
data['RECURSOS_HOGAR'] = (
    data['FAMI_TIENECOMPUTADOR'] +
    data['FAMI_TIENEINTERNET'] +
    data['FAMI_TIENEAUTOMOVIL'] +
    data['FAMI_TIENELAVADORA']
)
data['EDUCACION_PARENTAL'] = data[['FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE']].max(axis=1)
data['ESTRATO_X_RECURSOS'] = data['FAMI_ESTRATOVIVIENDA'] * data['RECURSOS_HOGAR']
data['TIENE_TECNOLOGIA'] = ((data['FAMI_TIENECOMPUTADOR'] == 1) &
                             (data['FAMI_TIENEINTERNET'] == 1)).astype(int)

features_finales = features + [
    'RECURSOS_HOGAR', 'EDUCACION_PARENTAL',
    'ESTRATO_X_RECURSOS', 'TIENE_TECNOLOGIA'
]


print(" Preparando entrenamiento y prueba...")
data_sample = data.sample(frac=0.5, random_state=42)
X = data_sample[features_finales]
y = data_sample[target]

X = X.fillna(X.median())
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n Entrenando modelo híbrido...")
lr = LinearRegression()
lr.fit(X_train, y_train)
residuos = y_train - lr.predict(X_train)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.07,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.5,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
xgb.fit(X_train, residuos)

pred_lineal = lr.predict(X_test)
correccion = xgb.predict(X_test)
y_pred_final = pred_lineal + correccion

r2 = r2_score(y_test, y_pred_final)
mae = mean_absolute_error(y_test, y_pred_final)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print("\n" + "="*60)
print(" RESULTADOS FINALES (25 VARIABLES)")
print("="*60)
print(f" R²: {r2:.4f}")
print(f" Porcentaje de explicación: {r2 * 100:.2f}%")
print(f" MAE: {mae:.3f} | RMSE: {rmse:.3f}")
print("="*60)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_final, alpha=0.4, color='teal')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.title("Comparación Real vs Predicho")
plt.xlabel("Valor Real (Puntaje promedio)")
plt.ylabel("Valor Predicho")
plt.tight_layout()
plt.show()


features_para_graficar = [f for f in features_finales if f != 'MOD_RAZONA_CUANTITAT_PUNT']

importancias = pd.DataFrame({
    'Variable': features_para_graficar,
    'Importancia': xgb.feature_importances_[:len(features_para_graficar)]
}).sort_values(by='Importancia', ascending=True)

plt.figure(figsize=(9, 7))
plt.barh(importancias['Variable'], importancias['Importancia'], color='skyblue')
plt.title("Importancia de Variables (XGBoost)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()


coef_df = pd.DataFrame({
    'Variable': features_para_graficar,
    'Coeficiente Lineal': lr.coef_[:len(features_para_graficar)]
}).sort_values(by='Coeficiente Lineal', ascending=False)

plt.figure(figsize=(9, 6))
plt.barh(coef_df['Variable'], coef_df['Coeficiente Lineal'], color='salmon')
plt.title("Coeficientes Lineales - Regresión Múltiple")
plt.xlabel("Valor del coeficiente")
plt.tight_layout()
plt.show()

print("\n Modelo completado correctamente.")
