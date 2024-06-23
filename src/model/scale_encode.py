import os

from scale_encode_utils import (
    dump_predictions,
    dump_preprocesser,
    fit_train,
    model_predict,
)

from ..intake.intake import clean_data, load_df, save_df

if __name__ == "__main__":
    regression_data_pth = (
        "../../data/flightdata/regression_flight_data_2022.csv"
    )
    classification_data_pth = (
        "../../data/flightdata/classification_flight_data_2022.csv"
    )

    BASE_FLIGHT_DATA_PTH = "../../data/flightdata/flights_2022.csv"

    # Define targets
    y_reg_target = ["departureDelayMinutes", "arrivalDelayMinutes"]
    y_clf_target = ["departureDelayBool", "arrivalDelayBool"]

    # REGRESSOR
    if os.path.exists(regression_data_pth):
        df = load_df(regression_data_pth)
        df = df.drop(columns=y_clf_target.append("flightNumber"), axis=1)

        y_reg = df[["departureDelayMinutes", "arrivalDelayMinutes"]]

        preprocessor, departure_model, arrival_model, X_test, y_test = (
            fit_train(df, y_reg, "regr")
        )
    else:
        df = load_df(BASE_FLIGHT_DATA_PTH)
        df = clean_data(df, proportion=0.4, balanced_target=False)
        save_df(df, regression_data_pth)

        df = df.drop(columns=y_clf_target.append("flightNumber"), axis=1)

        y_reg = df[["departureDelayMinutes", "arrivalDelayMinutes"]]

        preprocessor, departure_model, arrival_model, X_test, y_test = (
            fit_train(df, y_reg, "regr")
        )

    dump_preprocesser(preprocessor, "results/regr_preprocessor.joblib")

    y_preds, y_test, score = model_predict(
        departure_model, X_test, y_test, "regr"
    )

    dump_predictions(y_preds, y_test, "results/regr_predictions.csv")

    # CLASSIFIER
    if os.path.exists(classification_data_pth):
        df = load_df(classification_data_pth)
        df = df.drop(columns=y_reg_target.append("flightNumber"), axis=1)

        y_clf = df[["departureDelayBool", "arrivalDelayBool"]]

        preprocessor, departure_model, arrival_model, X_test, y_test = (
            fit_train(df, y_clf, "clf")
        )
    else:
        df = load_df(BASE_FLIGHT_DATA_PTH)
        df = clean_data(df, balanced_target=True)
        save_df(df, classification_data_pth)

        df = df.drop(columns=y_reg_target.append("flightNumber"), axis=1)

        y_clf = df[["departureDelayBool", "arrivalDelayBool"]]

        preprocessor, departure_model, arrival_model, X_test, y_test = (
            fit_train(df, y_clf, "clf")
        )

    dump_preprocesser(preprocessor, "results/clf_preprocessor.joblib")

    y_preds, y_test, score = model_predict(
        departure_model, X_test, y_test, "clf"
    )

    dump_predictions(y_preds, y_test, "results/clf_predictions.csv")
