import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
import os
import logging
from datetime import datetime


class LightGBMEstimatorWrapper(BaseEstimator):
    """Wrap a LightGBM Booster for sklearn-compatible usage."""

    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None, **kwargs):
        # No refit support for an already-trained Booster.
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1:
            return (proba > 0.5).astype(int)
        return proba.argmax(axis=1)

    def predict_proba(self, X):
        return self.booster.predict(X)

    def save_model(self, path):
        return self.booster.save_model(path)

    def feature_importance(self, **kwargs):
        return self.booster.feature_importance(**kwargs)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_MAP   = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
REVERSE_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}


# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logger(log_dir='../logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = os.path.join(log_dir, f'training_{timestamp}.log')

    logger = logging.getLogger('stellar')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # File handler — everything
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Console handler — INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                             datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger ready — log file: {log_file}")
    return logger


logger = setup_logger()


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(LABEL_MAP).values


def build_pseudo_labeled_training_data(X_train, y_train, X_pool, proba,
                                       class_thresholds=None):
    """Expand the training set with confident pseudo-labels from a probability matrix."""
    if class_thresholds is None:
        class_thresholds = {0: 0.995, 1: 0.995, 2: 0.95}

    predicted_class_idx = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)
    keep_mask = np.array([
        confidences[i] >= class_thresholds.get(int(predicted_class_idx[i]), 0.95)
        for i in range(len(confidences))
    ])

    if not keep_mask.any():
        return X_train.copy(), pd.Series(y_train, name='class').copy(), np.zeros(len(X_pool), dtype=bool), pd.Series([], dtype=object, name='class')

    X_pseudo = X_pool.loc[keep_mask].copy()
    pseudo_labels = np.array([REVERSE_MAP[i] for i in predicted_class_idx[keep_mask]])
    y_pseudo = pd.Series(pseudo_labels, index=X_pseudo.index, name='class')

    X_combined = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
    y_combined = pd.concat([
        pd.Series(y_train, name='class').reset_index(drop=True),
        y_pseudo.reset_index(drop=True),
    ], ignore_index=True)

    return X_combined, y_combined, keep_mask, y_pseudo


def pseudo_label_with_model(X_labeled, y_labeled, X_unlabeled, model,
                            confidence_threshold=0.95):
    """Expand the training set with confident pseudo-labels from an unlabeled pool."""
    probas = model.predict_proba(X_unlabeled)
    confidences = probas.max(axis=1)
    pseudo_labels = probas.argmax(axis=1)
    keep_mask = confidences >= confidence_threshold

    if not keep_mask.any():
        return X_labeled.copy(), y_labeled.copy(), pd.Index([], dtype=int)

    X_pseudo = X_unlabeled.loc[keep_mask].copy()
    y_pseudo = pd.Series(pseudo_labels[keep_mask], index=X_pseudo.index)

    X_combined = pd.concat([X_labeled, X_pseudo], axis=0).reset_index(drop=True)
    y_combined = pd.concat([pd.Series(y_labeled), y_pseudo], axis=0).reset_index(drop=True)
    return X_combined, y_combined, X_pseudo.index


def compute_permutation_importance(model, X: pd.DataFrame, y, n_repeats=10) -> pd.DataFrame:
    """Estimate feature importance with permutation importance."""
    if isinstance(model, lgb.Booster):
        model = LightGBMEstimatorWrapper(model)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring='balanced_accuracy',
    )
    return pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)


def explain_with_shap(model, X: pd.DataFrame, sample_size=200):
    """Return SHAP values when the optional shap dependency is available."""
    if shap is None:
        raise ImportError('shap is not installed. Install it with pip install shap.')

    sample = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return sample, shap_values


# ─────────────────────────────────────────────
#  OPTUNA TUNING
# ─────────────────────────────────────────────
def tune_lightgbm(X, y, n_trials=50):
    logger.info(f"Tuning LightGBM ({n_trials} trials)...")

    def objective(trial):
        params = {
            'objective':         'multiclass',
            'num_class':         3,
            'metric':            'multi_logloss',
            'verbosity':         -1,
            'n_jobs':            -1,
            'seed':              42,
            'is_unbalance':      True,
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 63, 255),
            'max_depth':         trial.suggest_int('max_depth', 6, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'feature_fraction':  trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            cb = [lgb.early_stopping(30, verbose=False),
                  lgb.log_evaluation(period=-1)]
            m = lgb.train(params, dtrain, num_boost_round=500,
                          valid_sets=[dval], callbacks=cb)
            preds = m.predict(X_val).argmax(axis=1)
            scores.append(balanced_accuracy_score(y_val, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best LightGBM score: {study.best_value:.4f}")
    logger.info(f"Best LightGBM params: {study.best_params}")
    return study.best_params


def tune_xgboost(X, y, n_trials=50):
    logger.info(f"Tuning XGBoost ({n_trials} trials)...")

    def objective(trial):
        params = {
            'objective':        'multi:softprob',
            'num_class':        3,
            'eval_metric':      'mlogloss',
            'verbosity':        0,
            'random_state':     42,
            'n_jobs':           -1,
            'n_estimators':     500,
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth':        trial.suggest_int('max_depth', 4, 10),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            m = xgb.XGBClassifier(**params, early_stopping_rounds=30,
                                   use_label_encoder=False)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = m.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best XGBoost score: {study.best_value:.4f}")
    logger.info(f"Best XGBoost params: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────
#  TRAIN FINAL MODELS
# ─────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    base = {
        'objective':    'multiclass',
        'num_class':    3,
        'metric':       'multi_logloss',
        'verbosity':    -1,
        'n_jobs':       -1,
        'seed':         42,
        'is_unbalance': True,
    }
    if params:
        base.update(params)

    # --- ORIGINAL: 'is_unbalance' only (silently a no-op for objective='multiclass' —
    # LightGBM only honors is_unbalance for 'binary' and 'multiclassova' objectives).
    # To revert: delete the two sample_weight lines below and change the two
    # lgb.Dataset(...) calls back to these:
    # dtrain = lgb.Dataset(X_train, label=y_train)
    # dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    # --- ACTIVE: proper per-sample class weighting for multiclass objective ---
    from sklearn.utils.class_weight import compute_sample_weight
    train_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    val_weight   = compute_sample_weight(class_weight='balanced', y=y_val)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=train_weight)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain, weight=val_weight)

    cb = [lgb.early_stopping(50, verbose=False),
          lgb.log_evaluation(period=100)]
    return lgb.train(base, dtrain, num_boost_round=2000,
                     valid_sets=[dval], callbacks=cb)


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    # Scale pos weight for STAR class
    n_galaxy = (y_train == 0).sum()
    n_star   = (y_train == 2).sum()
    spw = n_galaxy / max(n_star, 1)

    base = {
        'objective':          'multi:softprob',
        'num_class':          3,
        'eval_metric':        'mlogloss',
        'verbosity':          0,
        'random_state':       42,
        'n_jobs':             -1,
        'n_estimators':       2000,
        'early_stopping_rounds': 50,
        'use_label_encoder':  False,
    }
    if params:
        base.update(params)
    m = xgb.XGBClassifier(**base)
    m.fit(X_train, y_train,
          eval_set=[(X_val, y_val)], verbose=100)
    return m


def train_catboost(X_train, y_train, X_val, y_val):
    m = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        early_stopping_rounds=50,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=100,
    )
    m.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          use_best_model=True)
    return m


class TabularMLPClassifier:
    def __init__(self, input_dim, n_classes=3, hidden_dims=(128, 64), dropout=0.2, seed=42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        torch.manual_seed(seed)
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], n_classes),
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def _prepare_inputs(self, X, y=None):
        X_array = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_array)
        if y is None:
            return torch.from_numpy(X_scaled).to(self.device)
        y_array = np.asarray(y, dtype=np.int64)
        return torch.from_numpy(X_scaled).to(self.device), torch.from_numpy(y_array).to(self.device)

    @staticmethod
    def _focal_loss(logits, targets, alpha=0.25, gamma=2.0):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = -log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        loss = alpha * torch.pow(1 - p_t, gamma) * ce_loss
        return loss.mean()

    def fit(self, X, y, epochs=20, batch_size=256, verbose=False, label_smoothing=0.0, use_focal_loss=False):
        X_array = np.asarray(X, dtype=np.float32)
        self.scaler.fit(X_array)
        X_t, y_t = self._prepare_inputs(X_array, y)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        for epoch in range(epochs):
            running_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                if use_focal_loss:
                    loss = self._focal_loss(logits, yb)
                else:
                    loss = self.criterion(logits, yb)
                if label_smoothing > 0:
                    n_classes = logits.size(1)
                    smooth = torch.full_like(logits, label_smoothing / (n_classes - 1))
                    smooth.scatter_(1, yb.unsqueeze(1), 1.0 - label_smoothing)
                    loss = torch.sum(-smooth * torch.log_softmax(logits, dim=1), dim=1).mean()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)
            scheduler.step()
            if verbose:
                logger.info(f"MLP epoch {epoch+1}/{epochs} loss: {running_loss / len(dataset):.4f}")
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = self._prepare_inputs(X)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self


# ─────────────────────────────────────────────
#  CROSS VALIDATION
# ─────────────────────────────────────────────
def cross_validate_models(X, y_raw, n_splits=5,
                           lgb_params=None, xgb_params=None,
                           include_mlp=True):
    y   = encode_labels(y_raw)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros((len(X), 3))
    oof_xgb = np.zeros((len(X), 3))
    oof_cat = np.zeros((len(X), 3))
    oof_mlp = np.zeros((len(X), 3))
    lgb_models, xgb_models, cat_models, mlp_models = [], [], [], []

    logger.info(f"Starting {n_splits}-Fold CV | Train size: {len(X):,}")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"{'='*55}")
        logger.info(f"FOLD {fold+1}/{n_splits}")
        logger.info(f"{'='*55}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        logger.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,}")

        # LightGBM
        logger.info("Training LightGBM...")
        lgb_m = train_lightgbm(X_tr, y_tr, X_val, y_val, lgb_params)
        oof_lgb[val_idx] = lgb_m.predict(X_val)
        lgb_score = balanced_accuracy_score(y_val, oof_lgb[val_idx].argmax(1))
        logger.info(f"LightGBM BA: {lgb_score:.4f}")
        lgb_models.append(LightGBMEstimatorWrapper(lgb_m))

        # XGBoost
        logger.info("Training XGBoost...")
        xgb_m = train_xgboost(X_tr, y_tr, X_val, y_val, xgb_params)
        oof_xgb[val_idx] = xgb_m.predict_proba(X_val)
        xgb_score = balanced_accuracy_score(y_val, oof_xgb[val_idx].argmax(1))
        logger.info(f"XGBoost BA: {xgb_score:.4f}")
        xgb_models.append(xgb_m)

        # CatBoost
        logger.info("Training CatBoost...")
        cat_m = train_catboost(X_tr, y_tr, X_val, y_val)
        oof_cat[val_idx] = cat_m.predict_proba(X_val)
        cat_score = balanced_accuracy_score(y_val, oof_cat[val_idx].argmax(1))
        logger.info(f"CatBoost BA: {cat_score:.4f}")
        cat_models.append(cat_m)

        if include_mlp:
            logger.info("Training Tabular MLP...")
            mlp_m = TabularMLPClassifier(input_dim=X_tr.shape[1], n_classes=3, hidden_dims=(128, 64), dropout=0.2)
            mlp_m.fit(X_tr, y_tr, epochs=12, batch_size=256, verbose=False, label_smoothing=0.05, use_focal_loss=True)
            oof_mlp[val_idx] = mlp_m.predict_proba(X_val)
            mlp_score = balanced_accuracy_score(y_val, oof_mlp[val_idx].argmax(1))
            logger.info(f"MLP BA: {mlp_score:.4f}")
            mlp_models.append(mlp_m)

        # Fold summary
        if include_mlp:
            fold_ens = (oof_lgb[val_idx]*0.55 +
                        oof_xgb[val_idx]*0.25 +
                        oof_cat[val_idx]*0.10 +
                        oof_mlp[val_idx]*0.10)
        else:
            fold_ens = (oof_lgb[val_idx]*0.60 +
                        oof_xgb[val_idx]*0.35 +
                        oof_cat[val_idx]*0.05)
        fold_ens_score = balanced_accuracy_score(y_val, fold_ens.argmax(1))
        logger.info(f"Fold {fold+1} Ensemble BA: {fold_ens_score:.4f}")

    # Final scores
    if include_mlp:
        oof_ens = (oof_lgb*0.55 + oof_xgb*0.25 + oof_cat*0.10 + oof_mlp*0.10)
    else:
        oof_ens = (oof_lgb*0.60 + oof_xgb*0.35 + oof_cat*0.05)
    lgb_final = balanced_accuracy_score(y, oof_lgb.argmax(1))
    xgb_final = balanced_accuracy_score(y, oof_xgb.argmax(1))
    cat_final = balanced_accuracy_score(y, oof_cat.argmax(1))
    ens_final = balanced_accuracy_score(y, oof_ens.argmax(1))

    logger.info(f"{'='*55}")
    logger.info(f"FINAL CV RESULTS")
    logger.info(f"LightGBM  OOF: {lgb_final:.4f}")
    logger.info(f"XGBoost   OOF: {xgb_final:.4f}")
    logger.info(f"CatBoost  OOF: {cat_final:.4f}")
    if include_mlp:
        mlp_final = balanced_accuracy_score(y, oof_mlp.argmax(1))
        logger.info(f"MLP       OOF: {mlp_final:.4f}")
    logger.info(f"Ensemble  OOF: {ens_final:.4f}")
    logger.info(f"{'='*55}")

    return lgb_models, xgb_models, cat_models, mlp_models, oof_lgb, oof_xgb, oof_cat, oof_mlp


class MetaLearner:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)

    @staticmethod
    def _prepare_features(base_probabilities):
        arr = np.asarray(base_probabilities, dtype=float)
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f'Expected 2D or 3D probabilities, got shape {arr.shape}')

    def fit(self, base_probabilities, y):
        X_meta = self._prepare_features(base_probabilities)
        self.model.fit(X_meta, y)
        return self

    def predict_proba(self, base_probabilities):
        X_meta = self._prepare_features(base_probabilities)
        return self.model.predict_proba(X_meta)

    def predict(self, base_probabilities):
        X_meta = self._prepare_features(base_probabilities)
        return self.model.predict(X_meta)


def train_meta_learner(base_probabilities, y, random_state=42):
    return MetaLearner(random_state=random_state).fit(base_probabilities, y)


def save_models(lgb_models, xgb_models, cat_models, output_dir='models', mlp_models=None):
    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(lgb_models):
        m.save_model(f'{output_dir}/lgb_fold{i+1}.txt')
    for i, m in enumerate(xgb_models):
        m.save_model(f'{output_dir}/xgb_fold{i+1}.json')
    for i, m in enumerate(cat_models):
        m.save_model(f'{output_dir}/cat_fold{i+1}.cbm')
    if mlp_models is not None:
        for i, m in enumerate(mlp_models):
            m.save(f'{output_dir}/mlp_fold{i+1}.pt')
    logger.info(f"Saved {len(lgb_models)} LGB + "
                f"{len(xgb_models)} XGB + "
                f"{len(cat_models)} CAT models to '{output_dir}'")


# ─────────────────────────────────────────────
#  ADVERSARIAL VALIDATION
# ─────────────────────────────────────────────
def run_adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                n_splits: int = 5, threshold: float = 0.70) -> dict:
    """
    Detect distribution shift between train and test sets using adversarial validation.
    
    Combines train/test data with binary labels (0=train, 1=test) and trains a 
    stratified LightGBM classifier to measure if distributions are separable.
    Returns ROC-AUC score and feature importances indicating drift.
    
    The `threshold` parameter controls the AUC score above which data is considered
    drifted. A higher threshold (e.g., 0.70) is stricter; a lower threshold (e.g., 0.65)
    allows more distribution shift before raising an alert.
    
    Args:
        X_train: Training features (pd.DataFrame)
        X_test: Test features (pd.DataFrame)
        n_splits: Number of CV folds (default: 5)
        threshold: AUC threshold for drift detection (default: 0.70)
        
    Returns:
        dict with keys:
            - 'roc_auc': Out-of-fold ROC-AUC score (float)
            - 'feature_importances': DataFrame with drift scores per feature (pd.DataFrame)
            - 'is_drifted': Boolean flag indicating if AUC > threshold (bool)
    """
    from sklearn.metrics import roc_auc_score
    
    logger.info(f"Running Adversarial Validation | Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Drift Threshold: {threshold:.4f}")
    
    # Create binary classification task: 0=train, 1=test
    X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adversarial = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_test), dtype=int)
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_combined))
    feature_importance_sum = np.zeros(X_train.shape[1])
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_combined, y_adversarial)):
        X_tr, X_val = X_combined.iloc[tr_idx], X_combined.iloc[val_idx]
        y_tr, y_val = y_adversarial[tr_idx], y_adversarial[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_jobs': -1,
            'seed': 42,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'min_child_samples': 20,
        }
        
        cb = [lgb.early_stopping(30, verbose=False),
              lgb.log_evaluation(period=-1)]
        
        model = lgb.train(params, dtrain, num_boost_round=500,
                         valid_sets=[dval], callbacks=cb)
        
        oof_preds[val_idx] = model.predict(X_val)
        feature_importance_sum += model.feature_importance(importance_type='gain')
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        logger.info(f"Fold {fold+1}/{n_splits} Adversarial AUC: {fold_auc:.4f}")
    
    roc_auc = roc_auc_score(y_adversarial, oof_preds)
    feature_importance_avg = feature_importance_sum / n_splits
    
    # Create feature importance dataframe with explicit float casting to prevent
    # JSON serialization errors (TypeError: Object of type float64 is not JSON serializable)
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'drift_score': feature_importance_avg.astype(float),
    }).sort_values('drift_score', ascending=False).reset_index(drop=True)
    
    # Dynamically calculate is_drifted based on provided threshold
    is_drifted = bool(roc_auc > threshold)
    
    logger.info(f"Adversarial Validation OOF ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Distribution Shift Detected: {is_drifted} (threshold: {threshold:.4f})")
    logger.info(f"\nTop 10 Drifted Features:")
    logger.info(feature_importance_df.head(10).to_string(index=False))
    
    return {
        'roc_auc': float(roc_auc),
        'feature_importances': feature_importance_df,
        'is_drifted': is_drifted,
    }


# ─────────────────────────────────────────────
#  STACKING META-MODEL TRAINING
# ─────────────────────────────────────────────
def train_stacking_meta_model(oof_lgb: np.ndarray, oof_xgb: np.ndarray,
                               oof_cat: np.ndarray, y_train,
                               n_splits: int = 5) -> dict:
    """
    Train an enterprise-grade stacking meta-classifier using OOF predictions from base models.
    
    Takes out-of-fold probability arrays from LGB, XGB, and CAT models, stacks them
    as meta-features, and trains a cross-validated Ridge Classifier (Logistic Regression).
    Returns the trained meta-model and cross-validation metrics.
    
    Args:
        oof_lgb: OOF probabilities from LightGBM (n_samples, n_classes)
        oof_xgb: OOF probabilities from XGBoost (n_samples, n_classes)
        oof_cat: OOF probabilities from CatBoost (n_samples, n_classes)
        y_train: True labels (encoded as integers or raw labels)
        n_splits: Number of CV folds (default: 5)
        
    Returns:
        dict with keys:
            - 'meta_model': Trained LogisticRegression meta-classifier
            - 'cv_scores': Array of balanced accuracy scores per fold
            - 'mean_cv_score': Mean balanced accuracy across all folds
            - 'std_cv_score': Standard deviation of balanced accuracy
    """
    from sklearn.linear_model import LogisticRegression
    
    logger.info(f"Training Stacking Meta-Model | Train size: {len(y_train):,}")
    
    # Encode labels if necessary
    if isinstance(y_train, pd.Series):
        y_encoded = encode_labels(y_train)
    else:
        y_encoded = y_train
    
    # Stack base model predictions as meta-features
    X_meta = np.concatenate([oof_lgb, oof_xgb, oof_cat], axis=1)
    logger.info(f"Meta-feature shape: {X_meta.shape} (base models: LGB + XGB + CAT)")
    
    # Cross-validate meta-model
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y_encoded)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y_encoded[tr_idx], y_encoded[val_idx]
        
        # Train Ridge Logistic Regression for meta-model
        meta_model = LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            random_state=42,
        )
        meta_model.fit(X_tr, y_tr)
        
        preds = meta_model.predict(X_val)
        fold_score = balanced_accuracy_score(y_val, preds)
        cv_scores.append(fold_score)
        
        logger.info(f"Fold {fold+1}/{n_splits} Meta-Model BA: {fold_score:.4f}")
    
    cv_scores = np.array(cv_scores)
    
    # Train final meta-model on full data
    meta_model_final = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
    )
    meta_model_final.fit(X_meta, y_encoded)
    
    logger.info(f"{'='*55}")
    logger.info(f"Meta-Model CV Results:")
    logger.info(f"Mean BA: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"{'='*55}")
    
    return {
        'meta_model': meta_model_final,
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
    }


def save_meta_model(meta_model, output_path: str):
    """
    Save the trained meta-model using joblib.
    
    Args:
        meta_model: Trained LogisticRegression meta-classifier
        output_path: Path to save the model (.pkl file)
    """
    import joblib
    joblib.dump(meta_model, output_path)
    logger.info(f"Meta-model saved to: {output_path}")