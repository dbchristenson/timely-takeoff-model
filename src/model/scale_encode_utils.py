import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def dump_preprocesser(preprocessor, file_path: str):
    """Save a preprocessor object to a file."""
    dump(preprocessor, file_path)
    return


def dump_predictions(preds: list, test: list, file_path: str):
    """Save predictions and test values to a file."""
    df = pd.DataFrame({"predictions": preds, "test": test})
    df.to_csv(file_path, index=False)


def fit_train(df: pd.DataFrame, y: pd.DataFrame, model_type: str):
    """Creates a preprocessor from a given dataframe."""
    X = df.drop(columns=y, axis=1)

    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing for numeric and categorical data
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        "originCode",
        "destinationCode",
    ]  # Removed flightNumber

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", TargetEncoder(), categorical_features),
        ]
    )

    if model_type == "regr":
        model = LinearRegression()
    elif model_type == "clf":
        model = RandomForestClassifier()
    else:
        raise ValueError("Model type must be 'regr' or 'clf'. WTF u doin?")

    departure_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    arrival_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # Fit it up
    departure_model = departure_pipeline.fit(X_train, y_train[0])
    arrival_model = arrival_pipeline.fit(X_train, y_train[1])

    return preprocessor, departure_model, arrival_model, X_test, y_test


def model_predict(model, X_test, y_test, model_type: str):
    """Predicts the target values using a given model."""
    y_pred = model.predict(X_test)

    if model_type == "regr":
        # Calculate the RMSE
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {score}")
    elif model_type == "clf":
        # Calculate f1 score
        score = f1_score(y_test, y_pred)
        print(f"F1 Score: {score}")

    return y_pred, y_test, score
