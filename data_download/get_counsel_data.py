# File related library
import sys
from pathlib import Path

# 時間関連ライブラリ
import calendar

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# get utils
from utils.common_selenium_access_module import run_metabase  # noqa: E402, F401
from utils.output_date import get_current_date_string, get_current_date_parts, get_last_day_of_month, get_next_month_details  # noqa: E402, F401

# Usage
date_string = get_current_date_string()  # Current date in YYYYMMDD format
date_string = str(date_string)

# Get current date parts (year, month, day)
year, month, day = get_current_date_parts()
next_year, next_month, next_month_last_day = get_next_month_details()

last_day = get_last_day_of_month("current")
_, next_month_last_day = calendar.monthrange(next_year, next_month)


def main():
    TARGET_URL = "https://metabase.medical-frontier.net/question/7550-ck"
    FILE_PATTERN = r"~/Downloads/【ckカウンセリング予測】予測確認用データ_*.csv"
    DIRECTORY = r"D:/Forecast_future_conselling/"
    OUTPUT_FILE = rf"data/counseling_count.csv"
    run_metabase(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE)


if __name__ == "__main__":  # This is the correct way to check if the script is being run directly
    main()
