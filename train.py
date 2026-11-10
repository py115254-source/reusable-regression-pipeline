
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Configuration
DATA_FILE = "housing_data.csv"
MODEL_FILE = "regression_pipeline.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERICAL_FEATURES = ["Area", "Bedrooms", "Age"]
CATEGORICAL_FEATURES = ["Location"]
TARGET = "Price"


def load_data():
    """Load the housing dataset."""
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    return df


def create_pipeline():
    """Create preprocessing and regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate model using regression metrics."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Results")
    print("------------------------")
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    return predictions


def save_pipeline(model):
    """Save the trained pipeline using joblib."""

    joblib.dump(model, MODEL_FILE)
    print("\nModel saved successfully:", MODEL_FILE)


def load_pipeline():
    """Load the saved machine learning pipeline."""

    model = joblib.load(MODEL_FILE)
    print("Saved pipeline loaded successfully!")
    return model


def predict_new_data(model):
    """Generate predictions for new housing data."""

    new_data = pd.DataFrame({
        "Area": [1800, 1200],
        "Bedrooms": [3, 2],
        "Age": [5, 10],
        "Location": ["Urban", "Rural"]
    })

    predictions = model.predict(new_data)

    print("\nNew Data Predictions")
    print("--------------------")

    for index, prediction in enumerate(predictions):
        print(f"House {index + 1} Predicted Price: {prediction:.2f}")


def main():
    """Run the complete regression pipeline."""

    df = load_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = create_pipeline()

    model.fit(X_train, y_train)
    print("\nModel trained successfully!")

    evaluate_model(model, X_test, y_test)

    save_pipeline(model)

    loaded_model = load_pipeline()

    predict_new_data(loaded_model)


if __name__ == "__main__":
    main()
