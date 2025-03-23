import pandas as pd

read_csv = pd.read_csv("../data/counseling_count.csv")

group_df = read_csv.groupby("日付")["counseled"].sum().reset_index()
group_df.set_index("日付", inplace=True)

group_df.to_csv("../data/counseling_count_group.csv")