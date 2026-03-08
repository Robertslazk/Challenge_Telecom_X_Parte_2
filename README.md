# Challenge Telecom X - Parte 2

Proyecto de análisis y modelado predictivo para estimar la probabilidad de cancelación de clientes (`churn`) en Telecom X.

## Archivo principal

- `telecomx_parte2.ipynb`: notebook con todo el desarrollo del desafío.
- `telecomx_clean_relevante.csv`: dataset tratado usado en el modelado.

## Qué se desarrolló en `telecomx_parte2.ipynb`

### 1) Preparación inicial de datos

- Carga y validación del CSV.
- Revisión de calidad de datos (tipos, nulos, columnas irrelevantes).
- Codificación de variables categóricas con `OneHotEncoder`.
- Preparación de dos matrices:
  - `X_scaled`: para modelos sensibles a escala.
  - `X_tree`: para modelos basados en árboles.
- Diagnóstico de desbalance de clases:
  - `No`: 5174 (73.46%)
  - `Yes`: 1869 (26.54%)
  - Ratio mayoría/minoría: 2.77

### 2) Correlación y selección de variables

- Matriz de correlación para variables numéricas (incluyendo `churn_bin`).
- Revisión de variables más asociadas con cancelación.
- Análisis visual de relaciones clave:
  - `contract` x `churn`
  - `charges_total` x `churn`
  - `tenure` x `charges_total` coloreado por `churn`

### 3) Modelado predictivo

Se divide el dataset en entrenamiento/prueba con partición 80/20 y estratificación.

Modelos entrenados:

1. Regresión Logística (con normalización)
2. KNN (con normalización)
3. Árbol de Decisión (sin normalización)
4. Random Forest (sin normalización)

Métricas evaluadas:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Brecha `train-test` para revisión de overfitting/underfitting

### 4) Conclusiones

El notebook incluye una sección final con:

- Variables más influyentes según:
  - Coeficientes de Regresión Logística.
  - Importancias de Random Forest.
- Factores principales de cancelación.
- Recomendaciones estratégicas de retención basadas en resultados.

## Requisitos sugeridos

- Python 3.10+
- Jupyter Notebook / JupyterLab
- Librerías:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `nbformat` (solo si deseas manipular notebooks por script)

Instalación rápida:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Ejecución

```bash
jupyter notebook telecomx_parte2.ipynb
```

Ejecuta las celdas en orden para reproducir el análisis completo, el entrenamiento de modelos y las conclusiones.
