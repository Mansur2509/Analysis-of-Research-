"""
src/data_quality.py

screens the raw data before anyone trusts it. nothing here modifies
data, it just reports problems.
"""
import pandas as pd


def check_duplicates(df: pd.DataFrame, id_col: str = "Id") -> dict:
    n_dupes = df[id_col].duplicated().sum()
    return {"n_duplicates": n_dupes, "id_min": df[id_col].min(), "id_max": df[id_col].max()}


def check_range(df: pd.DataFrame, cols: list, expected_min=1, expected_max=5) -> pd.DataFrame:
    """flags any column where values fall outside the expected scale range."""
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        out_of_range = ((v < expected_min) | (v > expected_max)).sum()
        rows.append({"column": c, "min": v.min(), "max": v.max(), "out_of_range": out_of_range})
    return pd.DataFrame(rows)


def straightliners(df: pd.DataFrame, cols: list) -> pd.Series:
    """
    zero variance across a block = respondent picked the same answer
    for everything. doesn't catch someone who alternates 1-5-1-5 but
    that's a much rarer pattern than pure straightlining.
    """
    block = df[cols].apply(pd.to_numeric, errors="coerce")
    return block.std(axis=1)


def missingness_report(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False).head(top_n)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config as cfg

    pre = pd.read_excel(cfg.PRE_PATH)

    dupe_info = check_duplicates(pre)
    print("duplicate check:", dupe_info)

    m4_cols = [c for c in pre.columns if str(c).startswith("Q")
               and str(c)[1:4].isdigit() and 94 <= int(str(c)[1:4]) <= 118]
    range_check = check_range(pre, m4_cols[:6])
    print("\nrange check (sample):")
    print(range_check)

    sd = straightliners(pre, m4_cols)
    n_flat = (sd == 0).sum()
    print(f"\nstraightliners in module 4 block: {n_flat} ({n_flat/len(pre)*100:.1f}%), "
          f"min sd = {sd.min():.2f}")

    print("\nmissingness:")
    print(missingness_report(pre))
