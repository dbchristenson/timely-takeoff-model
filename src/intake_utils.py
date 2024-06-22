import datetime as dt

import pandas as pd


def convert_float_time(row: pd.Series) -> dt.datetime:
    """Converts a float time to a datetime object"""
    # convert flightDate value to string it is in format YYYYMMDD
    flight_date = str(row["flightDate"])

    to_dt_columns = [
        "scheduledDepartureTime",
        "actualDepartureTime",
        "scheduledArrivalTime",
        "actualArrivalTime",
        "wheelsOff",
        "wheelsOn",
    ]

    for column in to_dt_columns:
        # if the value is a datetime object, skip it
        if isinstance(row[column], dt.datetime):
            continue

        time = str(int(row[column])).zfill(4)
        hour = time[:2]
        minute = time[2:]

        if hour == "24":
            hour = "00"

        row[column] = dt.datetime.strptime(
            f"{flight_date} {hour}:{minute}", "%Y-%m-%d %H:%M"
        )

    return row


def cull_airport_codes(df: pd.DataFrame):
    kept_airport_codes = []

    code_cols = ["originCode", "destinationCode"]

    for col in code_cols:
        std = df[col].value_counts().std()
        avg = df[col].value_counts().mean()
        zscores = (df[col].value_counts() - avg) / std

        col_series = df[col].value_counts()
        col_series = col_series.reset_index()
        col_series["zscore"] = zscores.values

        # Keep codes with zscore greater than 0
        keep_codes = col_series[col_series["zscore"] > 0]["index"].to_list()

        kept_airport_codes.extend(keep_codes)

    # Keep unique codes
    kept_airport_codes = list(set(kept_airport_codes))

    # If the code of either the origin or destination is not
    # in the kept_airport_codes list, drop the row
    df = df[
        df["originCode"].isin(kept_airport_codes)
        & df["destinationCode"].isin(kept_airport_codes)
    ]

    return df


def yuh():
    return
