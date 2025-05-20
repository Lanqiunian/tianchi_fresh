import pandas as pd
import os


df_recall = "tianchi/data/2_processed/ranking_features_manual_test_target_20141219.parquet"

print(df_recall.head())

print(df_recall.columns)

print(df_recall.groupby('user_id').size().describe())