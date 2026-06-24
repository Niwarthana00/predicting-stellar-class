import pandas as pd
import numpy as np


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['g_i'] = df['g'] - df['i']
    df['r_z'] = df['r'] - df['z']
    df['u_r'] = df['u'] - df['r']
    df['g_z'] = df['g'] - df['z']
    return df


def add_band_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bands = ['u', 'g', 'r', 'i', 'z']
    df['band_mean']  = df[bands].mean(axis=1)
    df['band_std']   = df[bands].std(axis=1)
    df['band_min']   = df[bands].min(axis=1)
    df['band_max']   = df[bands].max(axis=1)
    df['band_range'] = df['band_max'] - df['band_min']
    return df


def add_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['redshift_log1p'] = np.log1p(df['redshift'].clip(lower=0))
    df['redshift_sq']    = df['redshift'] ** 2
    df['is_high_z']      = (df['redshift'] > 1.0).astype(int)
    df['is_star_z']      = (df['redshift'] < 0.01).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_color_indices(df)
    df = add_band_stats(df)
    df = add_redshift_features(df)
    return df


FEATURE_COLS = [
    'alpha', 'delta',
    'u', 'g', 'r', 'i', 'z', 'redshift',
    'u_g', 'g_r', 'r_i', 'i_z', 'g_i', 'r_z', 'u_r', 'g_z',
    'band_mean', 'band_std', 'band_min', 'band_max', 'band_range',
    'redshift_log1p', 'redshift_sq', 'is_high_z', 'is_star_z',
    'spectral_type_enc', 'galaxy_population_enc',
]