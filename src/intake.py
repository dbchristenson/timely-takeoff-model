import pandas as pd

from intake_utils import (
    convert_float_time,
    cull_airport_codes,
    scale_and_encode,
)

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


def clean_data(df: pd.DataFrame):
    """Perform data cleaning on a dataframe."""
    # Preliminary cleaning
    df = df.dropna()

    df = df.drop(columns=COLS_TO_DROP)

    df.columns = NEW_COLS

    # Convert some columns to the appropriate data types
    df = df.apply(convert_float_time, axis=1)

    # Remove less common airport codes to reduce cardinality
    df = cull_airport_codes(df)

    # Scale numerical columns and encode categorical columns
    df = scale_and_encode(df)

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
