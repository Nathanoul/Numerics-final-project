import pandas as pd

def shift_ids_df(df, columns):
    """
    Subtract 1 from selected ID columns in a DataFrame.
    """
    df = df.copy()
    for col in columns:
        df[col] -= 1
    return df