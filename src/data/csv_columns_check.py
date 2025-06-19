import pandas as pd

# feat_path の中身は先ほど実行時に表示されたパスをコピペしてください
feat_path = r"CC:\Projects\project_10k_to_1m_data\processed\USDJPY\M5\selfeat_USDJPY_M5_100000_20250620_043718.csv"

df_sample = pd.read_csv(feat_path, nrows=5)
print("Columns:", df_sample.columns.tolist())
print(df_sample.head())
