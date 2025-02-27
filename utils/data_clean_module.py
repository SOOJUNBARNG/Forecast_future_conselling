# Required library
import pandas as pd
from datetime import datetime


def df_clean(df, chunk_size=1000):
    """
    Cleans all columns in the given DataFrame efficiently by processing in chunks.
    """
    cleaned_chunks = []  # Store cleaned chunks

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]  # noqa E203

        # Apply transformations only to string columns
        chunk = chunk.applymap(
            lambda x: (
                x.strip()
                .replace("　", "")  # Remove full-width spaces
                .replace(r"\n", "")  # Remove literal newline characters
                .replace(" ", "")  # Remove regular spaces
                .replace(r"\s+", "")  # Remove any whitespace characters
                .replace(r"\n\d\s+", "")  # Remove specific pattern (e.g., newlines followed by digits and spaces)
                .replace("★", "")  # Remove specific character (★)
                .replace("\t", "")  # Remove tab characters
                if isinstance(x, str)
                else x
            )
        )

        cleaned_chunks.append(chunk)  # Append cleaned chunk

    return pd.concat(cleaned_chunks, ignore_index=True)  # Merge all chunks


def df_job_clean(df, Job):
    df = df.iloc[:, [0, 1] + list(range(8, df.shape[1]))]
    cols = list(df.columns)
    cols[:2] = ["Date", "Day"]
    df.columns = cols  # Assign back to DataFrame

    df = pd.melt(df, id_vars=["Date", "Day"], var_name="Clinic_name", value_name="Required_count")

    df["Job"] = Job

    return df


def parse_date(x):
    try:
        # Try to parse the date in the first format "%m/%d"
        return datetime.strptime(x, "%m/%d").day
    except ValueError:
        # If it fails, try the second format "%y%m/%d"
        return datetime.strptime(x, "%Y/%m/%d").day


def format_date(df, date_column, next_year, next_month, date_format="%Y%m%d"):
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    df = df[(df[date_column].dt.year == next_year) & (df[date_column].dt.month == next_month)]
    df[date_column] = df[date_column].dt.day  # Keep only the day
    return df


def split_date_year_month_day(x):
    year = int(str(x)[:4])  # Extract first 4 digits as year
    month = int(str(x)[4:6])  # Extract next 2 digits as month
    day = int(str(x)[6:])  # Extract last 2 digits as day
    return year, month, day
