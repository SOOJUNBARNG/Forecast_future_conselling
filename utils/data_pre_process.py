import pandas as pd

def get_nth_week_of_month(date):
    first_day_of_month = date.replace(day=1)
    first_weekday = first_day_of_month.weekday()  # Monday = 0, Sunday = 6
    current_weekday = date.weekday()
    
    # Calculate the nth week of the month
    nth_week = (date.day + first_weekday) // 7 + 1
    return nth_week

def data_process_by_clinic(df, data_start_date, current_date):
    # Load the dataset
    df = pd.read_csv("../data/counseling_count.csv")
    df = df[["クリニックID", "クリニック名", "日付", "counseled"]]
    df = df.rename(
        columns={
            "クリニックID": "clinic_id",
            "クリニック名": "clinic_name",
            "日付": "date",
        }
    )

    df = df[df["date"] > "2025-01-10"]

    df_clinic_unique = df[["clinic_id", "clinic_name"]].drop_duplicates()

    calendar_df = pd.read_csv("../data/calender.csv")
    calendar_df = calendar_df[["日付", "祝日フラグ", "TCB休診フラグ"]]
    calendar_df = calendar_df.rename(
        columns={
            "日付": "date",
            "祝日フラグ": "holiday_flag",
            "TCB休診フラグ": "tcb_holiday_flag",
        }
    )
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
    cross_df = df_clinic_unique.merge(calendar_df, how="cross")

    cross_df_counsel = cross_df.merge(
        df, on=["clinic_id", "clinic_name", "date"], how="left"
    )
    cross_df_counsel["counseled"] = cross_df_counsel["counseled"].fillna(0)

    # Display results
    print(cross_df_counsel.index[cross_df_counsel.index.duplicated()])
    # print(cross_df_counsel.columns)
    # print(cross_df_counsel.head(10))

    rest_day_df = pd.read_csv("../data/clinic_rest_day.csv")
    rest_day_df["name"] = rest_day_df["name"] + "院"
    rest_day_df = rest_day_df.rename(
        columns={
            "name": "clinic_name",
            "close_date": "date",
        }
    )

    df_calender_rest = pd.merge(
        cross_df_counsel, rest_day_df, on=["clinic_name", "date"], how="left"
    )
    df_calender_rest["status"] = df_calender_rest["status"].fillna(False)

    # Convert the date column to datetime format
    df_calender_rest["date"] = pd.to_datetime(df_calender_rest["date"])
    df_calender_rest = df_calender_rest[
        (df_calender_rest["date"] >= pd.Timestamp("2024-03-01"))
        &
        # (df_calender_rest["date"] <= pd.Timestamp(current_date) + pd.Timedelta(days=14))
        (
            df_calender_rest["date"]
            <= pd.to_datetime(f"{current_date}") + pd.Timedelta(days=28)
        )
    ]

    df_calender_rest = df_calender_rest.reset_index()
    # df_calender_rest.to_csv("check_for_great.csv", index=False)

    df_calender_rest["national_holiday"] = df_calender_rest["holiday_flag"].apply(
        lambda x: 0 if x is False else 1
    )
    df_calender_rest["tcb_holiday"] = df_calender_rest.apply(
        lambda row: (
            0 if row["tcb_holiday_flag"] is False and row["status"] is False else 1
        ),
        axis=1,
    )
    df_before_arima = df_calender_rest[
        [
            "clinic_id",
            "clinic_name",
            "date",
            "national_holiday",
            "tcb_holiday",
            "counseled",
        ]
    ]
    df_before_arima = df_before_arima.rename(
        columns={
            "tcb_holiday": "clinic_holiday",
        }
    )
    df_before_arima["day_of_week"] = df_before_arima["date"].dt.dayofweek
    df_before_arima["day_of_week"] = df_before_arima.apply(
        lambda x: 5 if x["national_holiday"] == 1 else x["day_of_week"], axis=1
    )
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )

    print(df_before_arima.index[df_before_arima.index.duplicated()])

    return df_before_arima


def data_process_group(df, data_start_date, current_date):
    # Load the dataset
    df = pd.read_csv("../data/counseling_count_group.csv")
    df = df[["日付", "counseled"]]
    df = df.rename(
        columns={
            "日付": "date",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df = df[df["date"] > f"{data_start_date}"]

    calendar_df = pd.read_csv("../data/calender.csv")
    calendar_df = calendar_df[["日付", "祝日フラグ", "TCB休診フラグ"]]
    calendar_df = calendar_df.rename(
        columns={
            "日付": "date",
            "祝日フラグ": "holiday_flag",
            "TCB休診フラグ": "tcb_holiday_flag",
        }
    )
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
    cross_df = df.merge(calendar_df, on="date", how="outer")
    cross_df["counseled"] = cross_df["counseled"].fillna(0)

    # cross_df.to_csv("print_check_ver2.csv", index=False)

    # Display results
    print(cross_df.index[cross_df.index.duplicated()])
    cross_df = cross_df.reset_index()

    cross_df["national_holiday"] = cross_df["holiday_flag"].apply(
        lambda x: 0 if x is False else 1
    )
    cross_df["tcb_holiday"] = cross_df.apply(
        lambda row: 0 if row["tcb_holiday_flag"] is False else 1, axis=1
    )
    data_process = cross_df[["date", "national_holiday", "tcb_holiday", "counseled"]]
    data_process["date"] = pd.to_datetime(data_process["date"], errors="coerce")
    data_process = data_process.rename(
        columns={
            "tcb_holiday": "clinic_holiday",
        }
    )
    data_process["day"] = data_process["date"].dt.dayofweek
    data_process["week_of_month"] = data_process["date"].apply(lambda x: get_nth_week_of_month(x))
    data_process["day"] = data_process.apply(
        lambda x: 5 if x["national_holiday"] == 1 else x["day"], axis=1
    )
    # df_before_sarima["day_of_week"] = df_before_sarima["date"].dt.dayofweek.map(
    #     {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    # )
    data_process = data_process[data_process["date"] >= pd.Timestamp(f"{data_start_date}")]

    print(data_process.index[data_process.index.duplicated()])

    return data_process