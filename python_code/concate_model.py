# 時間関連ライブラリ
import calendar
from datetime import datetime

# データ操作ライブラリ
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# プロジェクトルートを `sys.path` に追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# get utils
from utils.common_selenium_access_module import run_metabase  # noqa: E402, F401
from utils.output_date import get_current_date_string, get_current_date_parts, get_last_day_of_month, get_next_month_details  # noqa: E402, F401

current_date = datetime.today().date()

def get_current_plan_visit():
    TARGET_URL = "https://metabase.medical-frontier.net/question/5810-web"
    FILE_PATTERN = r"~/Downloads/web予約受付____予約作成日ベース　重複削除_　来院数予測_*.csv"
    DIRECTORY = r"D:/Forecast_future_conselling/"
    OUTPUT_FILE = rf"data/clinic_plan_visit.csv"
    run_metabase(TARGET_URL, FILE_PATTERN, DIRECTORY, OUTPUT_FILE)


def read_current_plan():
    # get_current_plan_visit()
    group_data = pd.read_csv("../data/clinic_plan_visit.csv")
    group_data["day"] = pd.to_datetime(group_data["day"])
    group_data = group_data.groupby("day", as_index=False)["reservations"].sum()
    group_data = group_data[["day", "reservations"]]
    group_data = group_data.rename(columns={"day": "Date", "reservations": "Counsel_plan"})
    
    return group_data


def get_date_youbi(target_datetime):
    day_of_week = target_datetime.strftime("%A")

    # Map English day to Japanese
    japanese_days = {
        "Monday": "月", "Tuesday": "火", "Wednesday": "水",
        "Thursday": "木", "Friday": "金", "Saturday": "土", "Sunday": "日"
    }
    return japanese_days[day_of_week]


def concate_model():
    get_arima_data = pd.read_csv("../data/arima_forecast.csv")
    get_arima_data = get_arima_data.rename(columns={
        "Forecast": "Arima_forcast",
        "Forecast 95% Top": "Top_Arima_forcast",
        "Forecast 95% Bot": "Bot_Arima_forcast",
    })
    # get_arima_data["Arima_forcast"] = get_arima_data["Arima_forcast"].astype(int)*-1

    get_sarima_data = pd.read_csv("../data/sarima_forecast.csv")
    get_sarima_data = get_sarima_data.rename(columns={
        "Forecast": "Sarima_forcast",
        "Forecast 95% Top": "Top_Sarima_forcast",
        "Forecast 95% Bot": "Bot_Sarima_forcast",
    })

    # `on="Date"` を指定してマージ
    get_total = get_arima_data.merge(get_sarima_data, on="Date", how="left")

    # `Date` カラムを datetime 型に変換
    get_total["Date"] = pd.to_datetime(get_total["Date"])
    get_total = get_total.merge(read_current_plan(), on="Date", how="left")

    # `Real_mid_line` を決定
    for index, row in get_total.iterrows():
        target_datetime = row["Date"]
        target_youbi = get_date_youbi(target_datetime)
        sarima_result = row["Sarima_forcast"]
        arima_result = row["Arima_forcast"]
        Top_Arima_forcast = row["Top_Arima_forcast"]
        Bot_Arima_forcast = row["Bot_Arima_forcast"]
        Top_Sarima_forcast = row["Top_Sarima_forcast"]
        Bot_Sarima_forcast = row["Bot_Sarima_forcast"]

        if target_youbi == "土":
            get_total.at[index, "Real_mid_line"] = sarima_result
            get_total.at[index, "Top_forcast"] = Top_Sarima_forcast
            get_total.at[index, "Bot_forcast"] = Bot_Sarima_forcast
        else:
            get_total.at[index, "Real_mid_line"] = arima_result
            get_total.at[index, "Top_forcast"] = Top_Arima_forcast
            get_total.at[index, "Bot_forcast"] = Bot_Arima_forcast  
                 

    return get_total


def show_graph(df):
    df = df.dropna()  # Remove NaN values
    
    # 日数制限を設定
    daycount_1month = 30
    daycount_1week = 7
    daycount_1day = 2

    # 現在の最新日を取得
    current_date = pd.to_datetime(datetime.today().date())
    # latest_date = df["Date"].max()
    # latest_date = latest_date.replace(day=1)

    # 各期間のフィルタ
    df_1month = df[df["Date"] <= current_date + pd.Timedelta(days=daycount_1month)]
    df_1week = df[df["Date"] <= current_date + pd.Timedelta(days=daycount_1week)]
    df_1day = df[df["Date"] <= current_date + pd.Timedelta(days=daycount_1day)]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Real_mid_line as a line chart
    ax1.plot(df["Date"], df["Real_mid_line"], label="Real_mid_line", color="blue", linewidth=2)

    # 1か月の範囲
    ax1.fill_between(df_1month["Date"], df_1month["Top_Arima_forcast"], df_1month["Bot_Arima_forcast"], 
                    alpha=0.3, color="lightblue", label="1month")

    # 1週間の範囲
    ax1.fill_between(df_1week["Date"], 
                    (df_1week["Real_mid_line"] + df_1week["Top_forcast"]) * 0.5, 
                    (df_1week["Real_mid_line"] + df_1week["Bot_forcast"]) * 0.5, 
                    alpha=0.5, color="lightcoral", label="1week")

    # 1日の範囲
    ax1.fill_between(df_1day["Date"], 
                    (df_1day["Real_mid_line"] * 2 + df_1day["Top_forcast"]) * 0.33, 
                    (df_1day["Real_mid_line"] * 2 + df_1day["Bot_forcast"]) * 0.33, 
                    alpha=0.7, color="green", label="1day")

    # Create a secondary y-axis for bar chart
    ax2 = ax1.twinx()
    ax2.bar(df["Date"], df["Counsel_plan"], color="gray", alpha=0.5, label="Counsel Plan", width=0.5)

    # Add text labels for Real_mid_line
    for i, row in df.iterrows():
        ax1.text(row["Date"], row["Real_mid_line"], int(row["Real_mid_line"]), 
                 fontsize=8, color="blue", ha="center", va="bottom", rotation=45)

    # Add text labels for Counsel_plan (bar chart)
    for i, row in df.iterrows():
        ax2.text(row["Date"], 0, int(row["Counsel_plan"]), 
                 fontsize=8, color="black", ha="center", va="bottom", rotation=45)

    # Labels and title
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Forecast Value", color="blue")
    ax2.set_ylabel("Counsel Plan", color="green")
    ax1.set_title("ARIMA vs SARIMA Forecast Comparison with Counsel Plan")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Grid
    ax1.grid(True)

    plt.show()


def main():
    # get_current_plan_visit()
    df = concate_model()
    df.to_csv("../data/total_data.csv", index=False)
    print(df.head())  # デバッグ用
    show_graph(df)  # グラフを表示


if __name__ == "__main__":
    main()
