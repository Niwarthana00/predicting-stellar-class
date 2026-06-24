import pandas as pd
import numpy as np


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sp_map = {'M': 0, 'A/F': 1, 'G/K': 2, 'O/B': 3}
    gp_map = {'Red_Sequence': 0, 'Blue_Cloud': 1}
    df['spectral_type_enc']     = df['spectral_type'].map(sp_map)
    df['galaxy_population_enc'] = df['galaxy_population'].map(gp_map)
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['spectral_type', 'galaxy_population'], errors='ignore')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_categoricals(df)
    df = drop_unused(df)
    return df