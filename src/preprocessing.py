import re
import pandas as pd

df = pd.read_excel("../data/raw/001_11jb_2025q1_20250807-213717.xlsx", header=None)

ROW_QUARTER = 2
ROW_FEATURE = 3
ROW_VALUE = 4

FEATURE_ALIASES = {
    "Reaalihintaindeksi": "reaalihintaindeksi",
    "Neliöhinta (EUR/m2)": "neliöhinta",
    "Keskipinta-ala m2": "keskipinta_ala",
    "Kauppojen lukumäärä": "kauppojen_lkm"
}

records = []
n_cols = df.shape[1]

for c in range(n_cols):
    q = df.iloc[ROW_QUARTER, c]
    if isinstance(q, str) and re.match(r"^\d{4}Q[1-4]$", q):
        row = {"quarter": q}
        block_cols = [c, c + 1, c + 2, c + 3]
        for k in block_cols:
            f_name = df.iloc[ROW_FEATURE, k]
            val = df.iloc[ROW_VALUE, k]
            if f_name in FEATURE_ALIASES:
                try:
                    row[FEATURE_ALIASES[f_name]] = float(val)
                except (ValueError, TypeError):
                    row[FEATURE_ALIASES[f_name]] = None
        if all(key in row and row[key] is not None for key in FEATURE_ALIASES.values()):
            records.append(row)

df_clean = pd.DataFrame(records)
df_clean["year"] = df_clean["quarter"].str[:4].astype(int)
df_clean["q_num"] = df_clean["quarter"].str[-1].astype(int)
df_clean = df_clean.sort_values(by=["year", "q_num"])

df_clean.to_csv("../data/processed/finnish_land_clean.csv", index=False)

