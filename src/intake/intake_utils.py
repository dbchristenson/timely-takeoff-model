import datetime as dt

import pandas as pd


def convert_float_time(row: pd.Series) -> dt.datetime:
    """Converts a float time to a datetime object"""
    try:
        # convert flightDate value to string it is in format YYYYMMDD
        flight_date = str(row["flightDate"].date())

        to_dt_columns = [
            "scheduledDepartureTime",
            "scheduledArrivalTime",
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
    except Exception as e:
        print(f"Skipping row due to error: {e}")

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
        "marketingAirlineCode",
        "flightNumberOperatingAirline",
        "flightNumberMarketingAirline",
        "fullAirlineName",
    ]

    # We keep operatingAirlineCode for encoding

    df = df.drop(
        columns=cols_to_drop,
    )
    return df


def calculate_operating_airline_reliability_score(df: pd.DataFrame):
    """Calculate a reliability score for each airline."""
    reliability_score = (
        df.groupby("operatingAirlineCode")[
            ["arrivalDelayMinutes", "departureDelayMinutes"]
        ]
        .mean()
        .sum(axis=1)
        / 2
    )

    # Create a dictionary to map the operatingAirlineCode to the score
    reliability_score_dict = reliability_score.to_dict()

    # Map the operatingAirlineCode to the reliability score
    df["reliabilityScore"] = df["operatingAirlineCode"].map(
        reliability_score_dict
    )

    # Drop the operatingAirlineCode column
    df = df.drop(columns=["operatingAirlineCode"])

    return df


def add_weather_data(df: pd.DataFrame):
    """
    Combine weather data about origin and destination weather during scheduled
    arrival and departure.
    """
    NEW_WEATHER_COLS = [
        "location_id",
        "time",
        "precipitation_mm",
        "rain_mm",
        "snowfall_mm",
        "weather_code",
        "cloud_cover_percent",
        "wind_speed_kmh",
        "wind_direction_degrees",
        "airport_code",
    ]

    # Load in iata/icao data
    airport_location_df = pd.read_csv("../../data/iata-icao.csv")

    # Clean
    cols_to_drop = ["country_code", "region_name", "icao", "airport"]
    airport_location_df = airport_location_df.drop(cols_to_drop, axis=1)

    # Only keep rows where value in iata column exists in
    # culled_df.originCode or culled_df.destinationCode
    airport_location_df = airport_location_df[
        airport_location_df["iata"].isin(df["originCode"])
        | airport_location_df["iata"].isin(df["destinationCode"])
    ]

    # Load historical weather data
    w_2022_df = pd.read_csv("../../data/weatherdata/weather_2022.csv")
    w_2022_loc_df = pd.read_csv("../../data/weatherdata/weather_locs_2022.csv")

    # Create mapper for airport codes to location_id
    location_dict = {}

    for x, y in zip(
        airport_location_df["iata"].values, w_2022_loc_df["location_id"].values
    ):
        location_dict[y] = x

    # Create columns in w_2022_df called airport_code
    w_2022_df["airport_code"] = w_2022_df["location_id"].map(location_dict)
    w_2022_df.columns = NEW_WEATHER_COLS

    # Convert time to datetime
    w_2022_df["time"] = pd.to_datetime(w_2022_df["time"])
    w_2022_df["hour"] = w_2022_df["time"].dt.hour
    w_2022_df["date"] = w_2022_df["time"].dt.date

    df.flightDate = df.flightDate.dt.date

    # Add hour column for scheduled departure and arrival time
    df["scheduledDepartureHour"] = df["scheduledDepartureTime"].dt.hour
    df["scheduledArrivalHour"] = df["scheduledArrivalTime"].dt.hour

    origin_weather = w_2022_df.copy()
    destination_weather = w_2022_df.copy()

    origin_cols = {
        col: "origin_" + col
        for col in origin_weather.columns
        if col not in ["airport_code", "date", "time", "hour"]
    }
    origin_weather.rename(columns=origin_cols, inplace=True)

    destination_cols = {
        col: "destination_" + col
        for col in destination_weather.columns
        if col not in ["airport_code", "date", "time", "hour"]
    }
    destination_weather.rename(columns=destination_cols, inplace=True)

    weather_df = df.merge(
        origin_weather,
        how="left",
        left_on=["originCode", "flightDate", "scheduledDepartureHour"],
        right_on=["airport_code", "date", "hour"],
    )

    weather_df = weather_df.merge(
        destination_weather,
        how="left",
        left_on=["destinationCode", "flightDate", "scheduledArrivalHour"],
        right_on=["airport_code", "date", "hour"],
    )

    origin_cols_to_drop = [
        "origin_location_id",
        "time_x",
        "airport_code_x",
        "hour_x",
        "date_x",
    ]

    dest_cols_to_drop = [
        "destination_location_id",
        "time_y",
        "airport_code_y",
        "hour_y",
        "date_y",
    ]

    df = weather_df.drop(columns=origin_cols_to_drop + dest_cols_to_drop)

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
