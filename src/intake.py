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
    # Path to the input Parquet file
    input_path = "path_to_input.parquet"

    # Path to the output Parquet file
    output_path = "path_to_output.parquet"

    # Load data
    data_table = load_parquet(input_path)

    # Clean data
    cleaned_data = clean_data(data_table)

    # Save cleaned data
    save_parquet(cleaned_data, output_path)
