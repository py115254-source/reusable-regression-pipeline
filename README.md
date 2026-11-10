
# Reusable Regression Pipeline

## Project Overview
This project demonstrates a reusable machine learning regression pipeline using Scikit-learn. It predicts housing prices using numerical and categorical features.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Key Features
- Data loading and preprocessing
- Numerical feature scaling using StandardScaler
- Categorical feature encoding using OneHotEncoder
- ColumnTransformer for combined preprocessing
- Linear Regression model
- Train-test split
- Model evaluation using MAE, MSE, RMSE and R2 Score
- Model serialization using Joblib
- Loading saved pipeline
- Predictions on new data

## Project Structure
```text
reusable-regression-pipeline/
│
├── housing_data.csv
├── train.py
├── requirements.txt
├── regression_pipeline.pkl
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python train.py
```

## Model Evaluation
The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Model Saving
The trained preprocessing and regression pipeline is saved using Joblib and can be loaded for future predictions.

## Conclusion
This project provides a reusable end-to-end regression pipeline that combines data preprocessing, model training, evaluation, saving, loading and prediction.
