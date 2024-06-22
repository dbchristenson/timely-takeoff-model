import datetime as dt

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLS_TO_DROP = [
    "Operated_or_Branded_Code_Share_Partners",
    "DOT_ID_Marketing_Airline",
    "DOT_ID_Operating_Airline",
    "IATA_Code_Marketing_Airline",
    "IATA_Code_Operating_Airline",
    "OriginAirportSeqID",
    "OriginCityMarketID",
    "DestAirportSeqID",
    "DestCityMarketID",
    "DepTimeBlk",
    "ArrDelayMinutes",
    "ArrivalDelayGroups",
    "ArrTimeBlk",
    "DistanceGroup",
    "DivAirportLandings",
    "DepDelayMinutes",
    "AirTime",
    "ActualElapsedTime",
    "Tail_Number",
    "DepartureDelayGroups",
    "OriginAirportID",
    "OriginCityName",
    "OriginState",
    "OriginStateFips",
    "OriginStateName",
    "OriginWac",
    "DestAirportID",
    "DestCityName",
    "DestState",
    "DestStateFips",
    "DestStateName",
    "DestWac",
    "Year",
    "Month",
    "DayofMonth",
    "Quarter",
    "Cancelled",
    "Diverted",
]

NEW_COLS = [
    "flightDate",
    "fullAirlineName",
    "originCode",
    "destinationCode",
    "scheduledDepartureTime",
    "actualDepartureTime",
    "departureDelayMinutes",
    "actualArrivalTime",
    "scheduledAirTime",
    "distanceMiles",
    "dayOfWeek",
    "marketingAirlineCode",
    "flightNumberMarketingAirline",
    "operatingAirlineCode",
    "flightNumberOperatingAirline",
    "departureDelayBool",
    "taxiOut",
    "wheelsOff",
    "wheelsOn",
    "taxiIn",
    "scheduledArrivalTime",
    "arrivalDelayMinutes",
    "arrivalDelayBool",
]


def load_df(file_path: str):
    """Load a csv file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def convert_float_time(row: pd.Series) -> dt.datetime:
    """Converts a float time to a datetime object"""
    # convert flightDate value to string it is in format YYYYMMDD
    flight_date = str(row["flightDate"])

    to_dt_columns = [
        "scheduledDepartureTime",
        "actualDepartureTime",
        "scheduledArrivalTime",
        "actualArrivalTime",
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


def clean_data(df: pd.DataFrame):
    """Perform data cleaning on a dataframe."""
    # Preliminary cleaning
    df = df.dropna()

    df = df.drop(columns=COLS_TO_DROP)

    df.columns = NEW_COLS

    # We drop more columns that are not useful, this is messy #hackathon
    redundant_cols = ["year", "month", "quarter", "dayOfMonth"]
    empty_cols = ["cancelledBool", "divertedBool"]  # values all emptyionWac",

    df = df.drop(
        columns=redundant_cols + empty_cols + origin_cols + destination_cols
    )

    # Convert some columns to the appropriate data types
    df = df.apply(convert_float_time, axis=1)

    return df


def save_df(df: pd.DataFrame, output_path: str):
    """Save a pandas DataFrame to a csv file."""
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    header_path = "../data/flightdata"
    years = ["2022"]
    in_paths = [f"{header_path}/flights_{y}.csv" for y in years]
    out_paths = [f"{header_path}/clean_flights_{y}.csv" for y in years]

    for in_path, out_path in zip(in_paths, out_paths):
        table = load_df(in_path)
        cleaned_table = clean_data(table)
        save_df(cleaned_table, out_path)

    print("Data cleaning complete!")
