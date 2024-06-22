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


def cull_airport_codes(df: pd.DataFrame, thresh: int = 1):
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
        keep_codes = col_series[col_series["zscore"] > thresh][col].to_list()

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


def cull_airlines(df: pd.DataFrame):
    """Remove airlines with fewer than 20000 flights."""
    airline_counts = df["fullAirlineName"].value_counts()
    airline_counts = airline_counts[airline_counts > 20000]

    df = df[df["fullAirlineName"].isin(airline_counts.index)]

    return df


def combine_airline_code_flight_number(df: pd.DataFrame):
    """Combine airline code and flight number into one column."""
    code = df["operatingAirlineCode"]
    number = df["flightNumberOperatingAirline"]

    df["flightNumber"] = code + number.astype(str)

    # Drop the original columns
    cols_to_drop = [
        "marketingAirlineNetwork",
        "flightNumberOperatingAirline",
        "flightNumberMarketingAirline",
        "fullAirlineName",
    ]

    # We keep operatingAirlineCode for encoding

    df = df.drop(
        columns=cols_to_drop,
    )
    return df


def combine_weather(df: pd.DataFrame):
    """
    Combine weather data about origin and destination weather during scheduled
    arrival and departure.
    """
    # load in iata/icao data
    airport_location_df = pd.read_csv("../data/iata-icao.csv")
    airport_location_df.head(3)

    # Clean
    airport_location_drops = ["country_code", "region_name", "icao", "airport"]
    airport_location_df = airport_location_df.drop(
        airport_location_drops, axis=1
    )

    # only keep rows where value in iata column exists in culled_df.originCode or culled_df.destinationCode
    airport_location_df = airport_location_df[
        airport_location_df["iata"].isin(df["originCode"])
        | airport_location_df["iata"].isin(df["destinationCode"])
    ]

    w_2022_df = pd.read_csv("../data/weatherdata/weather_2022.csv")
    w_2022_loc_df = pd.read_csv("../data/weatherdata/weather_locs_2022.csv")

    # Create mapper for airport codes to location_id
    location_dict = {}

    for x, y in zip(
        airport_location_df["iata"].values, w_2022_loc_df["location_id"].values
    ):
        location_dict[y] = x

    # create columns in w_2022_df and w_2022_loc_df called airport_code
    w_2022_df["airport_code"] = w_2022_df["location_id"].map(location_dict)
    w_2022_loc_df["airport_code"] = w_2022_loc_df["location_id"].map(
        location_dict
    )

    return df


def scale_and_encode(df: pd.DataFrame):
    """Scale numerical columns and encode categorical columns."""
    num_cols = df.select_dtypes(include="number").columns.to_list()
    cat_cols = df.select_dtypes(
        exclude=["number", "datetime"]
    ).columns.to_list()

    # Remove the target column for each model
    num_cols.remove("arrivalDelayMinutes")

    return df
