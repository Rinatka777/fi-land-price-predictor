import pandas as pd


df = pd.read_excel('../data/raw/001_11jb_2025q1_20250807-213717.xlsx'
)

quarters = df.iloc[1].dropna().tolist()
#test
print(quarters)


