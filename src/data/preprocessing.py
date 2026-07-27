import pandas as pd
import numpy as np


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ORIGINAL: hardcoded dict mapping ---
    # Uncomment these 4 lines, and comment out the fixed-order block below, to revert.
    sp_map = {'M': 0, 'A/F': 1, 'G/K': 2, 'O/B': 3}
    gp_map = {'Red_Sequence': 0, 'Blue_Cloud': 1}
    df['spectral_type_enc']     = df['spectral_type'].map(sp_map)
    df['galaxy_population_enc'] = df['galaxy_population'].map(gp_map)

    # --- ACTIVE: fixed-order pd.Categorical encoding (matches Notebook 05) ---
    # SPECTRAL_TYPES = ['A/F', 'G/K', 'M', 'O/B']
    # GALAXY_POPULATIONS = ['Blue_Cloud', 'Red_Sequence']
    # df['spectral_type_enc']     = pd.Categorical(df['spectral_type'], categories=SPECTRAL_TYPES).codes
    # df['galaxy_population_enc'] = pd.Categorical(df['galaxy_population'], categories=GALAXY_POPULATIONS).codes

    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=['spectral_type', 'galaxy_population'], errors='ignore')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_categoricals(df)
    df = drop_unused(df)
    return df