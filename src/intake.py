import pyarrow as pa
import pyarrow.parquet as pq


def load_parquet(file_path):
    """Load a Parquet file into a PyArrow Table."""
    return pq.read_table(file_path)


def clean_data(table):
    """Perform data cleaning on the PyArrow Table.

    Example: Remove rows where 'age' column is null or 'status' is inactive.
    """
    # Filter to remove rows with null 'age' or 'status' equals 'inactive'
    table = table.filter(
        (table["age"].is_valid()) & (table["status"] != "inactive")
    )
    return table


def save_parquet(table, output_path):
    """Save the cleaned PyArrow Table to a new Parquet file."""
    pq.write_table(table, output_path)


# Example usage
if __name__ == "__main__":

    header_path = "../data/flightdata"
    years = ["2018", "2019", "2022"]
    in_paths = [f"{header_path}/flights_{y}.parquet" for y in years]
    out_paths = [f"{header_path}/clean_flights_{y}.parquet" for y in years]

    for in_path, out_path in zip(in_paths, out_paths):
        table = load_parquet(in_path)
        cleaned_table = clean_data(table)
        save_parquet(cleaned_table, out_path)

    print("Data cleaning complete!")
