# Predicción de Ventas Globales de Videojuegos

Este repositorio contiene un proyecto de machine learning cuyo objetivo es predecir el éxito comercial de videojuegos a partir de datos históricos obtenidos de Kaggle. Se abordan desde el análisis exploratorio y el preprocesamiento hasta la optimización y selección de modelos. El modelo final, basado en Gradient Boosting por su alta precisión y robustez, se guarda para su implementación en producción.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características del Proyecto](#características-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción

Este proyecto utiliza técnicas de machine learning para predecir las ventas globales (en millones de copias) de videojuegos. A través del análisis de datos de Kaggle, se realiza un riguroso proceso de preprocesamiento, ingeniería de características, optimización de modelos y evaluación mediante RMSE. Tras la comparación de múltiples modelos, el mejor resultado se obtuvo con un modelo de **Gradient Boosting**, que ha sido guardado para producción y futura integración en sistemas de toma de decisiones.

## Características del Proyecto

- **Análisis Exploratorio de Datos (EDA):**  
  Se examina el dataset con histogramas y se aplican transformaciones logarítmicas para corregir la asimetría en las ventas.

- **Preprocesamiento e Ingeniería de Características:**  
  Tratamiento de valores nulos, transformación logarítmica de las ventas, codificación de variables categóricas y extracción de componentes temporales (año, mes, día, weekday).

- **Modelado y Optimización:**  
  Se entrenaron y optimizaron diversos modelos de regresión (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost y LightGBM) mediante GridSearchCV. Se evaluó el desempeño utilizando la métrica RMSE, que mide el error en las mismas unidades del target y penaliza fuertemente discrepancias grandes.

- **Implementación:**  
  El modelo final, que presentó el menor RMSE en validación cruzada, fue guardado para uso en producción mediante joblib.

## Estructura del Repositorio

Machine-Learning-Videogames/ ├── src/ │ ├── notebooks/ # Notebooks de análisis exploratorio y preprocesamiento. │ ├── scripts/ # Scripts para modelado y optimización. ├── models/ # Modelo final guardado para producción. ├── results_notebooks/ # Notebooks con resultados y visualizaciones. ├── README.md # Descripción del proyecto. └── requirements.txt 

# Dependencias necesarias.

## Requisitos

- Python 3.x
- Librerías: NumPy, pandas, scikit-learn, seaborn, matplotlib, xgboost, lightgbm, catboost, joblib

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/Machine-Learning-Videogames.git
