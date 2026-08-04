import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import os

REVERSE_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}


class LGBWrapper:

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        return self.booster.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def feature_importance(self, **kwargs):
        return self.booster.feature_importance(**kwargs)


def wrap_lgb_model(booster):
    return LGBWrapper(booster)


def load_lgb_models(model_dir='models'):
    models = []
    if not os.path.exists(model_dir):
        print(f"Not found: {model_dir}")
        return models
    for f in sorted(os.listdir(model_dir)):
        full_path = os.path.join(model_dir, f)
        if os.path.isfile(full_path) and f.startswith('lgb_fold') and f.endswith('.txt'):
            booster = lgb.Booster(model_file=full_path)

            models.append(wrap_lgb_model(booster))
    print(f"Loaded {len(models)} LightGBM from '{model_dir}'")
    return models


def load_xgb_models(model_dir='models'):
    models = []
    if not os.path.exists(model_dir):
        return models
    for f in sorted(os.listdir(model_dir)):
        full_path = os.path.join(model_dir, f)
        if os.path.isfile(full_path) and f.startswith('xgb_fold') and f.endswith('.json'):
            m = xgb.XGBClassifier()
            m.load_model(full_path)
            models.append(m)
    print(f"Loaded {len(models)} XGBoost from '{model_dir}'")
    return models


def load_cat_models(model_dir='models'):
    models = []
    if not os.path.exists(model_dir):
        return models
    for f in sorted(os.listdir(model_dir)):
        full_path = os.path.join(model_dir, f)
        if os.path.isfile(full_path) and f.startswith('cat_fold') and f.endswith('.cbm'):
            m = CatBoostClassifier()
            m.load_model(full_path)
            models.append(m)
    print(f"Loaded {len(models)} CatBoost from '{model_dir}'")
    return models


def predict_proba_ensemble(X: pd.DataFrame,
                            lgb_models, xgb_models,
                            cat_models=None,
                            w_lgb=0.60, w_xgb=0.35, w_cat=0.05) -> np.ndarray:
    lgb_preds = np.zeros((len(X), 3))
    xgb_preds = np.zeros((len(X), 3))

    for m in lgb_models:
        if isinstance(m, lgb.Booster):
            m = wrap_lgb_model(m)
        lgb_preds += m.predict_proba(X)
    lgb_preds /= len(lgb_models)

    for m in xgb_models:
        xgb_preds += m.predict_proba(X)
    xgb_preds /= len(xgb_models)

    if cat_models and len(cat_models) > 0:
        cat_preds = np.zeros((len(X), 3))
        for m in cat_models:
            cat_preds += m.predict_proba(X)
        cat_preds /= len(cat_models)
        ensemble = (lgb_preds*w_lgb + xgb_preds*w_xgb + cat_preds*w_cat)
    else:
        total_w  = w_lgb + w_xgb
        ensemble = (lgb_preds*(w_lgb/total_w) + xgb_preds*(w_xgb/total_w))

    return ensemble


def predict_classes(X: pd.DataFrame,
                    lgb_models, xgb_models,
                    cat_models=None,
                    w_lgb=0.60, w_xgb=0.35, w_cat=0.05) -> np.ndarray:
    proba = predict_proba_ensemble(
        X, lgb_models, xgb_models, cat_models, w_lgb, w_xgb, w_cat)
    return np.array([REVERSE_MAP[i] for i in proba.argmax(axis=1)])


def make_submission_frame(test_ids, proba, output_path=None, index=False):
    preds = np.array([REVERSE_MAP[i] for i in proba.argmax(axis=1)])
    sub = pd.DataFrame({'id': test_ids, 'class': preds})
    if output_path is not None:
        sub.to_csv(output_path, index=index)
    return sub


def predict_single(feature_dict: dict,
                   lgb_models, xgb_models,
                   cat_models, feature_cols: list,
                   w_lgb=0.60, w_xgb=0.35, w_cat=0.05) -> dict:
    from src.data.preprocessing import preprocess
    from src.features.engineering import build_features

    df = pd.DataFrame([feature_dict])
    df = preprocess(df)
    df = build_features(df)
    X  = df[feature_cols]

    proba      = predict_proba_ensemble(
        X, lgb_models, xgb_models, cat_models, w_lgb, w_xgb, w_cat)[0]
    pred_class = REVERSE_MAP[proba.argmax()]

    return {
        'predicted_class': pred_class,
        'probabilities': {
            'GALAXY': round(float(proba[0]), 4),
            'QSO':    round(float(proba[1]), 4),
            'STAR':   round(float(proba[2]), 4),
        }
    }


# ─────────────────────────────────────────────
#  TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
def predict_with_tta(lgb_models, xgb_models, cat_models,
                      X: pd.DataFrame,
                      noise_level: float = 0.01,
                      num_repeats: int = 5,
                      w_lgb: float = 0.60,
                      w_xgb: float = 0.35,
                      w_cat: float = 0.05,
                      random_state: int = 42) -> np.ndarray:
    """
    Generate robust predictions using Test-Time Augmentation (TTA) for tabular data.
    
    Applies minor Gaussian noise to numerical features across multiple iterations,
    generates predictions for each augmented copy, and averages the results.
    This technique reduces variance and improves generalization on test data.
    
    Args:
        lgb_models: List of trained LightGBM models
        xgb_models: List of trained XGBoost models
        cat_models: List of trained CatBoost models
        X: Test features (pd.DataFrame)
        noise_level: Standard deviation of Gaussian noise (default: 0.01)
        num_repeats: Number of augmentation iterations (default: 5)
        w_lgb: Weight for LightGBM predictions (default: 0.60)
        w_xgb: Weight for XGBoost predictions (default: 0.35)
        w_cat: Weight for CatBoost predictions (default: 0.05)
        random_state: Random seed for reproducibility
        
    Returns:
        Ensemble probability array (n_samples, 3) averaged across all augmentations
    """
    np.random.seed(random_state)
    
    # Identify numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Accumulate predictions across all augmentations (including original)
    accumulated_proba = np.zeros((len(X), 3))
    
    # Original predictions (no noise)
    original_proba = predict_proba_ensemble(X, lgb_models, xgb_models, cat_models,
                                           w_lgb=w_lgb, w_xgb=w_xgb, w_cat=w_cat)
    accumulated_proba += original_proba
    
    # Augmented predictions with noise
    for repeat in range(num_repeats):
        X_augmented = X.copy()
        
        # Convert numeric columns to float before adding noise to avoid dtype issues.
        if numerical_cols:
            X_augmented[numerical_cols] = X_augmented[numerical_cols].astype(float)
            noise = np.random.normal(loc=0.0, scale=noise_level,
                                    size=(len(X_augmented), len(numerical_cols)))
            X_augmented[numerical_cols] += noise
        
        # Get predictions for augmented data
        augmented_proba = predict_proba_ensemble(X_augmented, lgb_models, xgb_models, 
                                               cat_models, w_lgb=w_lgb, w_xgb=w_xgb, 
                                               w_cat=w_cat)
        accumulated_proba += augmented_proba
    
    # Average predictions across all iterations (original + num_repeats)
    tta_proba = accumulated_proba / (1 + num_repeats)
    
    return tta_proba


# ─────────────────────────────────────────────
#  META-MODEL LOADING
# ─────────────────────────────────────────────
def load_meta_model(model_path: str):
    """
    Load a trained meta-model from disk.
    
    Args:
        model_path: Path to the saved meta-model (.pkl file)
        
    Returns:
        Trained LogisticRegression meta-classifier
    """
    import joblib
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Meta-model not found at: {model_path}")
    
    meta_model = joblib.load(model_path)
    print(f"Meta-model loaded from: {model_path}")
    return meta_model


def predict_with_meta_model(meta_model,
                            oof_lgb: np.ndarray,
                            oof_xgb: np.ndarray,
                            oof_cat: np.ndarray) -> np.ndarray:
    """
    Generate stacking meta-model predictions from base model OOF probabilities.
    
    Args:
        meta_model: Trained LogisticRegression meta-classifier
        oof_lgb: OOF probabilities from LightGBM (n_samples, n_classes)
        oof_xgb: OOF probabilities from XGBoost (n_samples, n_classes)
        oof_cat: OOF probabilities from CatBoost (n_samples, n_classes)
        
    Returns:
        Stacking meta-model probability predictions (n_samples, 3)
    """
    # Stack base model predictions as meta-features
    X_meta = np.concatenate([oof_lgb, oof_xgb, oof_cat], axis=1)
    
    # Generate meta-predictions
    meta_proba = meta_model.predict_proba(X_meta)
    
    return meta_proba