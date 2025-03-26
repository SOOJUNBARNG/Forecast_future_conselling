import pandas as pd

# Load CSV file
ad_df = pd.read_csv("../data/advertisement_data.csv")

# Group by 予約受付日 and 来院日, summing the 広告費 column
ad_df_group = ad_df.groupby(["予約受付日"], as_index=False)[["広告費"]].sum()
ad_df_group["広告費"] = ad_df_group["広告費"].astype(int)
ad_df_group = ad_df_group[ad_df_group["広告費"] != 0]

# Save the grouped data
ad_df_group.to_csv("../data/advertisement_group_data.csv", index=False)
