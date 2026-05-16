<div align="center">

# 🚗 Used Car Price Prediction

### Interactive Web Application powered by XGBoost

**Computer Engineering Final Degree Project**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)


### 🌐 [**Try the App Live**](https://pablomartint-tfg.streamlit.app)

</div>

---

## 📌 Overview

This project is the deployment-ready demonstration of a **Computer Engineering Final Degree Project** focused on predicting **used car prices** using machine learning. An XGBoost regressor was trained on a dataset of approximately **746,587 real listings** scraped from `Cars.com`, achieving an **R² of 95.62%** and a mean absolute error of **$2,447**.

The model is exposed through an interactive web application built with **Streamlit**. Users introduce the characteristics of a vehicle (model, mileage, age, engine size, transmission, etc.) and obtain an instant price estimation along with comparative cards that contextualize the prediction against the market average for that specific model.

---

## 🎯 Key Features

- **Real-time price prediction** with a single click.
- **Smart input restrictions** by manufacturer (e.g. Tesla → electric only, Land Rover → automatic only).
- **Auto-selection** of single-option fields (e.g. Tesla automatically sets fuel type to electric).
- **Dynamic model dropdown** that updates based on the selected manufacturer.
- **Comparative dashboard** showing how the predicted price, mileage and age relate to the model's market average.
- **Reliability warnings** when input values fall outside the dataset's training range.

---

## 📊 Model Performance

The XGBoost regressor was selected as the winning model after benchmarking against six alternatives, including Linear Regression, Random Forest, Gradient Boosting, LightGBM, and a Multi-Layer Perceptron neural network. Final results on the test set:

<div align="center">

| Metric | Value |
|:---:|:---:|
| **R² (Coefficient of Determination)** | **95.62%** |
| **MAE (Mean Absolute Error)** | **$2,447** |
| **MAPE (Mean Absolute Percentage Error)** | **8.09%** |
| **Number of features** | 13 |
| **Training set size** | 522,610 |
| **Validation set size** | 111,988 |
| **Test set size** | 111,989 |
| **Unique vehicle models in dataset** | 4,453 |

</div>

---

## 🧠 How the Model Works

1. **Data Ingestion** — Each input introduced by the user is transformed into a 13-dimensional feature vector matching the model's training schema.
2. **Categorical Encoding** — Categorical variables are encoded using ordinal encoding. The vehicle model uses target encoding based on the average standardized log-price observed during training.
3. **Prediction** — The XGBoost regressor outputs a standardized log-price prediction. This value is first de-standardized using the training mean and standard deviation, and then converted back to dollars via the exponential function.
4. **Confidence Interval** — A symmetric ±8.09% margin (matching the model's MAPE) communicates the natural uncertainty of the estimation.
5. **Comparative Analysis** — Three reference cards show how the predicted price, mileage and age position the vehicle relative to the model's average in the training set.

---

## 📂 Project Structure

- `app.py` — Main Streamlit application.
- `xgb_opt.pkl` — Trained XGBoost model.
- `app_artifacts/` — Encoders, reference dictionaries and target standardization parameters.
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python runtime version.
- `.gitignore`
- `README.md`

---

## 🚀 Run Locally

Clone the repository, install the dependencies and launch the app:

    git clone https://github.com/PabloMartinTejedor/Used-Car-Price-Prediction-App-Computer-Engineering-Final-Degree-Project.git
    cd Used-Car-Price-Prediction-App-Computer-Engineering-Final-Degree-Project
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

The application will open automatically in your default browser at `http://localhost:8501`.

---

## ⚠️ Disclaimer

This application is an **academic demonstration**. Predictions are based on a public dataset of US-based listings (`Cars.com`, April 2023) and may not generalize accurately to vehicles outside that market or sold under different economic conditions.