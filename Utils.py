import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV"""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    """Basic cleaning & encoding"""
    df = df.dropna()
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df
