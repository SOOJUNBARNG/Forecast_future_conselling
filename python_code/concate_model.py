# 時間関連ライブラリ
from datetime import datetime

# データ操作ライブラリ
import pandas as pd
import matplotlib.pyplot as plt


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

    # `Real_mid_line` を決定
    for index, row in get_total.iterrows():
        target_datetime = row["Date"]
        target_youbi = get_date_youbi(target_datetime)
        sarima_result = row["Sarima_forcast"]
        arima_result = row["Arima_forcast"]

        if target_youbi == "土":
            get_total.at[index, "Real_mid_line"] = sarima_result
        else:
            get_total.at[index, "Real_mid_line"] = arima_result

    return get_total


def show_graph(df):
    df = df.dropna()  # NaN を削除
    plt.figure(figsize=(12, 6))
    
    # 実際の値のライン
    plt.plot(df["Date"], df["Real_mid_line"], label="Real_mid_line", color="blue")

    # 予測区間の塗りつぶし
    plt.fill_between(df["Date"], df["Top_Arima_forcast"], df["Bot_Arima_forcast"], alpha=0.3, color="red", label="Arima Range")

    plt.xlabel("Date")
    plt.ylabel("Forecast Value")
    plt.title("ARIMA vs SARIMA Forecast Comparison")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    df = concate_model()
    df.to_csv("../data/total_data.csv", index=False)
    print(df.head())  # デバッグ用
    show_graph(df)  # グラフを表示


if __name__ == "__main__":
    main()
