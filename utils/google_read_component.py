# 時間関連library
import time

# ファイル関連 library
import pandas as pd

# Google関連 library
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def column_letter(N):
    result = ""
    while N > 0:
        n, remainder = divmod(N - 1, 26)
        result = chr(65 + remainder) + result
    return result


def connect_gspread(json, key, worksheet_name):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json, scope)
    gc = gspread.authorize(credentials)
    workbook = gc.open_by_key(key)
    worksheet_name = f"{worksheet_name}"

    try:
        # Try to open the existing sheet
        worksheet = workbook.worksheet(worksheet_name)
        # print(f"Connected to existing worksheet '{
        #       worksheet_name}' successfully!")
        return worksheet
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{worksheet_name}' not found")
        return None  # Or you can add logic to create a new sheet if needed


def read_data_in_googlespread_sheet(
    json, sheet_key, worksheet_name, header_row=None, data_start_row=None
):
    """
    Reads data from Google Sheets and converts it into a DataFrame.

    :param json: Path to the service account JSON file for Google Sheets API
    :param sheet_key: Google Sheets key (part of the sheet URL)
    :param worksheet_name: Name of the worksheet to fetch data from
    :param header_row: Row number where headers are located (1-based index). Default is 1 if not specified.
    :param data_start_row: Row number where data starts (1-based index). Default is 2 if not specified.
    :return: DataFrame containing the sheet data
    """
    print("Connecting to Google Sheets...")
    time.sleep(5)

    # JSON file and spreadsheet key
    json = f"{json}"
    spread_sheet_key = f"{sheet_key}"
    ws = connect_gspread(json, spread_sheet_key, worksheet_name)

    # Read all values in the sheet
    data = ws.get_all_values()  # This retrieves all rows and columns

    # Set default values if None is passed
    if header_row is None:
        header_row = 0
    if data_start_row is None:
        data_start_row = 1

    # Adjust for custom header and data start points
    headers = data[
        header_row
    ]  # The header starts from `header_row`, adjusted for 0-based index
    values = data[
        data_start_row:
    ]  # The data starts from `data_start_row`, adjusted for 0-based index

    # Convert to DataFrame
    df = pd.DataFrame(values, columns=headers)
    print("Data read successfully from Google Sheet!")
    return df


def put_data_into_googlespread_sheet(
    df, json, sheet_key, worksheet_name, next_month_youbi
):
    # print("Connecting to Google Sheets...")

    time.sleep(5)
    # JSON file and spreadsheet key
    spread_sheet_key = f"{sheet_key}"
    ws = connect_gspread(json, spread_sheet_key, worksheet_name)

    # Convert the DataFrame to a list of lists
    header = df.columns.tolist()  # Get header row
    next_month_youbi_row = [
        "",
        "",
        "",
        "",
        "",
    ] + next_month_youbi  # Add two blank columns for "Clinic" and "Staff Name"
    df_values = [header] + [next_month_youbi_row] + df.values.tolist()

    # Calculate the number of columns and construct the cell range
    num_columns = df.shape[1]
    last_column_letter = column_letter(num_columns)  # Get the last column letter
    cell_range = (
        f"A1:{last_column_letter}{len(df) + 2}"  # Calculate the range to update
    )

    ws.update(cell_range, df_values)
