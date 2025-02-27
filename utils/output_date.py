from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar


def get_current_date_string():
    """Returns the current date in YYYYMMDD format."""
    return datetime.today().strftime("%Y%m%d")


def get_one_day_earlier_date_string():
    """Returns the date of the previous day in YYYYMMDD format."""
    return (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")


def get_current_datetime():
    """Returns the current datetime."""
    return datetime.now()


def get_current_date_parts():
    """Returns a tuple of current year, month, and day."""
    current_datetime = get_current_datetime()
    return current_datetime.year, current_datetime.month, current_datetime.day


def get_last_day_of_month(options_="current"):
    """
    Returns the last day of the specified month.
    options_ can be:
        - "current" (default): last day of the current month
        - "next": last day of the next month
    """
    year, month, _ = get_current_date_parts()

    if options_ == "next":
        # Get the year and month for the next month
        next_month_datetime = get_current_datetime() + relativedelta(months=1)
        next_month_year = next_month_datetime.year
        next_month = next_month_datetime.month
        return int(calendar.monthrange(next_month_year, next_month)[1])

    # Default to current month if no valid option is provided
    return int(calendar.monthrange(year, month)[1])


def get_next_month_details():
    """Returns the year, month, day, and last day of the next month."""
    next_month_datetime = get_current_datetime() + relativedelta(months=1)
    next_month_year = next_month_datetime.year
    next_month = next_month_datetime.month
    next_month_day = next_month_datetime.day
    return next_month_year, next_month, next_month_day


def get_previous_month_details():
    """Returns the year, month, day, and last day of the next month."""
    previous_month_datetime = get_current_datetime() + relativedelta(months=-1)
    previous_month_year = previous_month_datetime.year
    previous_month = previous_month_datetime.month
    previous_month_day = previous_month_datetime.day
    return previous_month_year, previous_month, previous_month_day


def get_next_month_youbi():
    """Returns a list of days of the week (in Japanese) for the next month."""
    next_month_year, next_month, next_month_last_day = get_next_month_details()
    next_month_last_day = get_last_day_of_month("next")

    next_month_youbi = []
    for i in range(1, next_month_last_day + 1):
        target_datetime = datetime(next_month_year, next_month, i)
        day_of_week = target_datetime.strftime("%A")

        # Map English day to Japanese
        japanese_days = {"Monday": "月", "Tuesday": "火", "Wednesday": "水", "Thursday": "木", "Friday": "金", "Saturday": "土", "Sunday": "日"}
        next_month_youbi.append(japanese_days[day_of_week])

    return next_month_youbi
