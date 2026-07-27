import pandas as pd
import numpy as np


BAND_COLS = ['u', 'g', 'r', 'i', 'z']
COLOR_COLS = ['u_g', 'g_r', 'r_i', 'i_z', 'g_i', 'r_z', 'u_r', 'g_z', 'u_i', 'u_z', 'r_g']


def _prepare_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in BAND_COLS + ['redshift']:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors='coerce')

    for col in BAND_COLS:
        if col in prepared.columns:
            prepared[col] = prepared[col].fillna(prepared[col].median())

    if 'redshift' in prepared.columns:
        prepared['redshift'] = prepared['redshift'].fillna(prepared['redshift'].median())

    return prepared


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_numeric_frame(df).copy()

    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['g_i'] = df['g'] - df['i']
    df['r_z'] = df['r'] - df['z']
    df['u_r'] = df['u'] - df['r']
    df['g_z'] = df['g'] - df['z']
    df['u_i'] = df['u'] - df['i']
    df['u_z'] = df['u'] - df['z']
    df['r_g'] = df['r'] - df['g']

    df['g_div_z'] = df['g'] / (df['z'].replace(0, np.nan) + 1e-6)
    df['r_div_i'] = df['r'] / (df['i'].replace(0, np.nan) + 1e-6)
    df['u_div_r'] = df['u'] / (df['r'].replace(0, np.nan) + 1e-6)
    df['g_div_r'] = df['g'] / (df['r'].replace(0, np.nan) + 1e-6)

    df['gz_x_ri'] = df['g_z'] * df['r_i']
    df['gr_x_iz'] = df['g_r'] * df['i_z']

    for col in COLOR_COLS + ['g_div_z', 'r_div_i', 'u_div_r', 'g_div_r', 'gz_x_ri', 'gr_x_iz']:
        df[col] = _clean_numeric_series(df[col])

    return df


def add_band_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bands = df[BAND_COLS]

    df['band_mean'] = bands.mean(axis=1)
    df['band_std'] = bands.std(axis=1, ddof=0)
    df['band_min'] = bands.min(axis=1)
    df['band_max'] = bands.max(axis=1)
    df['band_range'] = df['band_max'] - df['band_min']
    df['band_sum'] = bands.sum(axis=1)
    df['band_skew'] = bands.skew(axis=1).fillna(0.0)
    df['band_median'] = bands.median(axis=1)
    df['band_iqr'] = bands.quantile(0.75, axis=1) - bands.quantile(0.25, axis=1)
    df['band_cv'] = (df['band_std'] / (df['band_mean'].replace(0, np.nan) + 1e-6)).fillna(0.0)

    for col in ['band_mean', 'band_std', 'band_min', 'band_max', 'band_range', 'band_sum', 'band_skew', 'band_median', 'band_iqr', 'band_cv']:
        df[col] = _clean_numeric_series(df[col])

    return df


def add_redshift_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    redshift = pd.to_numeric(df['redshift'], errors='coerce').fillna(df['redshift'].median())

    df['redshift_log1p'] = np.log1p(redshift.clip(lower=0))
    df['redshift_sq'] = redshift ** 2
    df['redshift_sqrt'] = np.sqrt(redshift.clip(lower=0))
    df['redshift_abs'] = redshift.abs()
    df['is_high_z'] = (redshift > 1.0).astype(int)
    df['is_star_z'] = (redshift < 0.01).astype(int)
    df['is_redshift_zero'] = (redshift.abs() < 0.0001).astype(int)
    df['is_very_high_z'] = (redshift > 3.0).astype(int)
    df['is_negative_z'] = (redshift < 0).astype(int)

    df['redshift_x_gz'] = redshift * df['g_z']
    df['redshift_x_bandstd'] = redshift * df['band_std']
    df['redshift_x_bandmean'] = redshift * df['band_mean']
    df['redshift_over_bandmean'] = (redshift / (df['band_mean'].replace(0, np.nan) + 1e-6)).fillna(0.0)
    df['g_r_x_redshift'] = df['g_r'] * redshift
    df['u_g_x_redshift'] = df['u_g'] * redshift

    for col in ['redshift_log1p', 'redshift_sq', 'redshift_sqrt', 'redshift_abs', 'is_high_z', 'is_star_z', 'is_redshift_zero', 'is_very_high_z', 'is_negative_z', 'redshift_x_gz', 'redshift_x_bandstd', 'redshift_x_bandmean', 'redshift_over_bandmean', 'g_r_x_redshift', 'u_g_x_redshift']:
        df[col] = _clean_numeric_series(df[col])

    return df


def add_misc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = pd.to_numeric(df['delta'], errors='coerce').fillna(df['delta'].median())
    df['delta_abs'] = delta.abs()
    df['delta_abs'] = _clean_numeric_series(df['delta_abs'])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_color_indices(df)
    df = add_band_stats(df)
    df = add_redshift_features(df)
    df = add_misc_features(df)

    for col in df.columns:
        if col not in ['class', 'spectral_type', 'galaxy_population']:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = _clean_numeric_series(df[col])

    return df


FEATURE_COLS = [
    'alpha', 'delta', 'delta_abs',
    'u', 'g', 'r', 'i', 'z', 'redshift',
    # Color indices
    'u_g', 'g_r', 'r_i', 'i_z', 'g_i', 'r_z', 'u_r', 'g_z',
    'u_i', 'u_z', 'r_g',
    # Ratios & products
    'g_div_z', 'r_div_i', 'u_div_r', 'g_div_r',
    'gz_x_ri', 'gr_x_iz',
    # Band stats
    'band_mean', 'band_std', 'band_min', 'band_max',
    'band_range', 'band_sum', 'band_skew', 'band_median', 'band_iqr', 'band_cv',
    # Redshift features
    'redshift_log1p', 'redshift_sq', 'redshift_sqrt', 'redshift_abs',
    'is_high_z', 'is_star_z', 'is_redshift_zero',
    'is_very_high_z', 'is_negative_z', 'redshift_x_gz', 'redshift_x_bandstd', 'redshift_x_bandmean', 'redshift_over_bandmean',
    'g_r_x_redshift', 'u_g_x_redshift',
    # Encoded categoricals
    'spectral_type_enc', 'galaxy_population_enc',
]