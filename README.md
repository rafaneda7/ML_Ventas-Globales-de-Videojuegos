# Global Video Game Sales Prediction

This repository contains a machine learning project that aims to predict the commercial success of video games based on historical data obtained from Kaggle. The project covers everything from exploratory analysis and preprocessing to optimization and model selection. The final model, based on Gradient Boosting for its high accuracy and robustness, is saved for production deployment.

---

## Table of Contents

- [Description](#description)
- [Project Features](#project-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Use](#use)
- [Contributions](#contributions)
- [License](#license)

---

## Description

This project uses machine learning techniques to predict global sales (in millions of copies) of video games. Through Kaggle data analysis, a rigorous process of preprocessing, feature engineering, model optimization, and RMSE evaluation is performed. After comparing multiple models, the best result was obtained with a **Gradient Boosting** model, which has been saved for production and future integration into decision-making systems.

---

## Project Features

- **Exploratory Data Analysis (EDA):**
The dataset is examined with histograms and logarithmic transformations are applied to correct for skewed sales.

- **Preprocessing and Feature Engineering:**
Null value handling, logarithmic transformation of sales, coding of categorical variables, and extraction of temporal components (year, month, day, weekday).

- **Modeling and Optimization:**
Several regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, and LightGBM) were trained and optimized using GridSearchCV. Performance was evaluated using the RMSE metric, which measures error in the same units as the target and heavily penalizes large discrepancies.

- **Implementation:**
The final model, which showed the lowest RMSE in cross-validation, was saved for production use using joblib.

---

## Requirements

- Python 3.12.7
- Libraries:
`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `joblib`

---

## Installation

1. Clone the repository:
   ```bash
  git clone https://github.com/your_user/Machine-Learning-Videogames.git

2. Change to the project directory:
   ```bash
  cd Machine-Learning-Videogames

3. Install dependencies:
   ```bash
  pip install -r requirements.txt

---

## Usage

- **The notebooks for exploratory analysis and preprocessing are available at:**
  ```bash
src/notebooks/

- **To run the optimization, evaluation, and save the final model, use the scripts in:**
  ```bash
src/scripts/

- **The final model (Gradient Boosting) is saved in:**
  ```bash
src/models/best_model_GradientBoosting.pkl

---

## Contributions

Contributions are welcome!

Follow these steps:

1. Fork the repository.

2. Create a branch with your new feature:
   ```bash
git checkout -b new-feature

3. Make your changes and commit:
   ```bash
git commit -am 'Add Feature'

4. Send a Pull Request to the original repository:

---

## License

This project is licensed under the MIT License (LICENSE).

---

# Predicción de Ventas Globales de Videojuegos

Este repositorio contiene un proyecto de machine learning cuyo objetivo es predecir el éxito comercial de videojuegos a partir de datos históricos obtenidos de Kaggle. Se abordan desde el análisis exploratorio y el preprocesamiento hasta la optimización y selección de modelos. El modelo final, basado en Gradient Boosting por su alta precisión y robustez, se guarda para su implementación en producción.

---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Características del Proyecto](#características-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Descripción

Este proyecto utiliza técnicas de machine learning para predecir las ventas globales (en millones de copias) de videojuegos. A través del análisis de datos de Kaggle, se realiza un riguroso proceso de preprocesamiento, ingeniería de características, optimización de modelos y evaluación mediante RMSE. Tras la comparación de múltiples modelos, el mejor resultado se obtuvo con un modelo de **Gradient Boosting**, que ha sido guardado para producción y futura integración en sistemas de toma de decisiones.

---

## Características del Proyecto

- **Análisis Exploratorio de Datos (EDA):**  
  Se examina el dataset con histogramas y se aplican transformaciones logarítmicas para corregir la asimetría en las ventas.

- **Preprocesamiento e Ingeniería de Características:**  
  Tratamiento de valores nulos, transformación logarítmica de las ventas, codificación de variables categóricas y extracción de componentes temporales (año, mes, día, weekday).

- **Modelado y Optimización:**  
  Se entrenaron y optimizaron diversos modelos de regresión (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost y LightGBM) mediante GridSearchCV. Se evaluó el desempeño utilizando la métrica RMSE, que mide el error en las mismas unidades del target y penaliza fuertemente discrepancias grandes.

- **Implementación:**  
  El modelo final, que presentó el menor RMSE en validación cruzada, fue guardado para uso en producción mediante joblib.

---

## Requisitos

- Python 3.12.7
- Librerías:  
  `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `joblib`

---

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/Machine-Learning-Videogames.git

2. Entra en el directorio del proyecto:
   ```bash
   cd Machine-Learning-Videogames

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

---

## Uso

- **Los notebooks para análisis exploratorio y preprocesamiento están disponibles en:**  
   ```bash
   src/notebooks/

- **Para ejecutar la optimización, evaluación y guardar el modelo final, usa los scripts en:**  
   ```bash
   src/scripts/

- **El modelo final (Gradient Boosting) está guardado en:**  
   ```bash
   src/models/best_model_GradientBoosting.pkl

---

## Contribuciones

¡Las contribuciones son bienvenidas!

Sigue estos pasos:

1. Haz un fork del repositorio.
   
2. Crea una rama con tu nueva funcionalidad:
   ```bash
   git checkout -b nueva-funcionalidad

3. Realiza los cambios y realiza un commit:
   ```bash
   git commit -am 'Añadir funcionalidad'

4. Envía un Pull Request al repositorio original:

---

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).

