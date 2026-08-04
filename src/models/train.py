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
    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        predicted_probabilities = self.predict_proba(X)
        if predicted_probabilities.ndim == 1:
            return (predicted_probabilities > 0.5).astype(int)
        return predicted_probabilities.argmax(axis=1)

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
except ImportError:
    shap = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_MAP   = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
REVERSE_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}


def setup_logger(log_directory='../logs'):
    os.makedirs(log_directory, exist_ok=True)
    timestamp_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_directory, f'training_{timestamp_string}.log')

    stellar_logger = logging.getLogger('stellar')
    stellar_logger.setLevel(logging.DEBUG)
    stellar_logger.handlers = []

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                             datefmt='%H:%M:%S')
    file_handler.setFormatter(log_formatter)
    console_handler.setFormatter(log_formatter)

    stellar_logger.addHandler(file_handler)
    stellar_logger.addHandler(console_handler)

    stellar_logger.info(f"Logger ready — log file: {log_file_path}")
    return stellar_logger


logger = setup_logger()


def encode_labels(label_series: pd.Series) -> np.ndarray:
    return label_series.map(LABEL_MAP).values


def build_pseudo_labeled_training_data(labeled_features, labeled_targets, unlabeled_pool_features, pool_class_probabilities,
                                       class_thresholds=None):
    if class_thresholds is None:
        class_thresholds = {0: 0.995, 1: 0.995, 2: 0.95}

    predicted_class_indices = np.argmax(pool_class_probabilities, axis=1)
    max_class_confidences = np.max(pool_class_probabilities, axis=1)
    confident_sample_mask = np.array([
        max_class_confidences[i] >= class_thresholds.get(int(predicted_class_indices[i]), 0.95)
        for i in range(len(max_class_confidences))
    ])

    if not confident_sample_mask.any():
        return labeled_features.copy(), pd.Series(labeled_targets, name='class').copy(), np.zeros(len(unlabeled_pool_features), dtype=bool), pd.Series([], dtype=object, name='class')

    pseudo_labeled_features = unlabeled_pool_features.loc[confident_sample_mask].copy()
    pseudo_label_strings = np.array([REVERSE_MAP[i] for i in predicted_class_indices[confident_sample_mask]])
    pseudo_label_series = pd.Series(pseudo_label_strings, index=pseudo_labeled_features.index, name='class')

    combined_features = pd.concat([labeled_features, pseudo_labeled_features], axis=0).reset_index(drop=True)
    combined_targets = pd.concat([
        pd.Series(labeled_targets, name='class').reset_index(drop=True),
        pseudo_label_series.reset_index(drop=True),
    ], ignore_index=True)

    return combined_features, combined_targets, confident_sample_mask, pseudo_label_series


def pseudo_label_with_model(labeled_features, labeled_targets, unlabeled_features, model,
                            confidence_threshold=0.95):
    predicted_probabilities = model.predict_proba(unlabeled_features)
    max_confidences = predicted_probabilities.max(axis=1)
    predicted_label_indices = predicted_probabilities.argmax(axis=1)
    confident_sample_mask = max_confidences >= confidence_threshold

    if not confident_sample_mask.any():
        return labeled_features.copy(), labeled_targets.copy(), pd.Index([], dtype=int)

    confident_pseudo_features = unlabeled_features.loc[confident_sample_mask].copy()
    confident_pseudo_labels = pd.Series(predicted_label_indices[confident_sample_mask], index=confident_pseudo_features.index)

    combined_features = pd.concat([labeled_features, confident_pseudo_features], axis=0).reset_index(drop=True)
    combined_targets = pd.concat([pd.Series(labeled_targets), confident_pseudo_labels], axis=0).reset_index(drop=True)
    return combined_features, combined_targets, confident_pseudo_features.index


def compute_permutation_importance(model, feature_matrix: pd.DataFrame, target_labels, n_repeats=10) -> pd.DataFrame:
    if isinstance(model, lgb.Booster):
        model = LightGBMEstimatorWrapper(model)

    permutation_result = permutation_importance(
        model,
        feature_matrix,
        target_labels,
        n_repeats=n_repeats,
        random_state=42,
        scoring='balanced_accuracy',
    )
    return pd.DataFrame({
        'feature': feature_matrix.columns,
        'importance_mean': permutation_result.importances_mean,
        'importance_std': permutation_result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)


def explain_with_shap(model, feature_matrix: pd.DataFrame, sample_size=200):
    if shap is None:
        raise ImportError('shap is not installed. Install it with pip install shap.')

    sampled_features = feature_matrix.sample(min(sample_size, len(feature_matrix)), random_state=42)
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(sampled_features)
    return sampled_features, shap_values


def tune_lightgbm(feature_matrix, target_labels, n_trials=50):
    logger.info(f"Tuning LightGBM ({n_trials} trials)...")

    def objective(trial):
        lightgbm_params = {
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
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        for train_indices, val_indices in stratified_kfold.split(feature_matrix, target_labels):
            train_features, val_features = feature_matrix.iloc[train_indices], feature_matrix.iloc[val_indices]
            train_targets, val_targets = target_labels[train_indices], target_labels[val_indices]
            train_dataset = lgb.Dataset(train_features, label=train_targets)
            val_dataset   = lgb.Dataset(val_features, label=val_targets, reference=train_dataset)
            callbacks = [lgb.early_stopping(30, verbose=False),
                  lgb.log_evaluation(period=-1)]
            trained_model = lgb.train(lightgbm_params, train_dataset, num_boost_round=500,
                          valid_sets=[val_dataset], callbacks=callbacks)
            val_predictions = trained_model.predict(val_features).argmax(axis=1)
            fold_scores.append(balanced_accuracy_score(val_targets, val_predictions))
        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best LightGBM score: {study.best_value:.4f}")
    logger.info(f"Best LightGBM params: {study.best_params}")
    return study.best_params


def tune_xgboost(feature_matrix, target_labels, n_trials=50):
    logger.info(f"Tuning XGBoost ({n_trials} trials)...")

    def objective(trial):
        xgboost_params = {
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
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        for train_indices, val_indices in stratified_kfold.split(feature_matrix, target_labels):
            train_features, val_features = feature_matrix.iloc[train_indices], feature_matrix.iloc[val_indices]
            train_targets, val_targets = target_labels[train_indices], target_labels[val_indices]
            trained_model = xgb.XGBClassifier(**xgboost_params, early_stopping_rounds=30,
                                   use_label_encoder=False)
            trained_model.fit(train_features, train_targets, eval_set=[(val_features, val_targets)], verbose=False)
            val_predictions = trained_model.predict(val_features)
            fold_scores.append(balanced_accuracy_score(val_targets, val_predictions))
        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best XGBoost score: {study.best_value:.4f}")
    logger.info(f"Best XGBoost params: {study.best_params}")
    return study.best_params


def train_lightgbm(train_features, train_targets, val_features, val_targets, hyperparameter_overrides=None):
    lightgbm_base_params = {
        'objective':    'multiclass',
        'num_class':    3,
        'metric':       'multi_logloss',
        'verbosity':    -1,
        'n_jobs':       -1,
        'seed':         42,
        'is_unbalance': True,
    }
    if hyperparameter_overrides:
        lightgbm_base_params.update(hyperparameter_overrides)

    from sklearn.utils.class_weight import compute_sample_weight
    train_sample_weights = compute_sample_weight(class_weight='balanced', y=train_targets)
    val_sample_weights   = compute_sample_weight(class_weight='balanced', y=val_targets)

    train_dataset = lgb.Dataset(train_features, label=train_targets, weight=train_sample_weights)
    val_dataset   = lgb.Dataset(val_features,   label=val_targets, reference=train_dataset, weight=val_sample_weights)

    callbacks = [lgb.early_stopping(50, verbose=False),
          lgb.log_evaluation(period=100)]
    return lgb.train(lightgbm_base_params, train_dataset, num_boost_round=2000,
                     valid_sets=[val_dataset], callbacks=callbacks)


def train_xgboost(train_features, train_targets, val_features, val_targets, hyperparameter_overrides=None):
    galaxy_class_count = (train_targets == 0).sum()
    star_class_count   = (train_targets == 2).sum()
    star_scale_pos_weight = galaxy_class_count / max(star_class_count, 1)

    xgboost_base_params = {
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
    if hyperparameter_overrides:
        xgboost_base_params.update(hyperparameter_overrides)
    trained_model = xgb.XGBClassifier(**xgboost_base_params)
    trained_model.fit(train_features, train_targets,
          eval_set=[(val_features, val_targets)], verbose=100)
    return trained_model


def train_catboost(train_features, train_targets, val_features, val_targets):
    trained_model = CatBoostClassifier(
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
    trained_model.fit(train_features, train_targets,
          eval_set=(val_features, val_targets),
          use_best_model=True)
    return trained_model


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

    def _prepare_inputs(self, raw_features, raw_targets=None):
        feature_array = np.asarray(raw_features, dtype=np.float32)
        scaled_feature_array = self.scaler.transform(feature_array)
        if raw_targets is None:
            return torch.from_numpy(scaled_feature_array).to(self.device)
        target_array = np.asarray(raw_targets, dtype=np.int64)
        return torch.from_numpy(scaled_feature_array).to(self.device), torch.from_numpy(target_array).to(self.device)

    @staticmethod
    def _focal_loss(logits, targets, alpha=0.25, gamma=2.0):
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probabilities)
        cross_entropy_loss = -log_probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        target_class_probabilities = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_loss_value = alpha * torch.pow(1 - target_class_probabilities, gamma) * cross_entropy_loss
        return focal_loss_value.mean()

    def fit(self, raw_features, raw_targets, epochs=20, batch_size=256, verbose=False, label_smoothing=0.0, use_focal_loss=False):
        feature_array = np.asarray(raw_features, dtype=np.float32)
        self.scaler.fit(feature_array)
        feature_tensor, target_tensor = self._prepare_inputs(feature_array, raw_targets)
        tensor_dataset = TensorDataset(feature_tensor, target_tensor)
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        for epoch in range(epochs):
            epoch_running_loss = 0.0
            for batch_features, batch_targets in data_loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_features)
                if use_focal_loss:
                    loss = self._focal_loss(logits, batch_targets)
                else:
                    loss = self.criterion(logits, batch_targets)
                if label_smoothing > 0:
                    num_classes = logits.size(1)
                    smoothed_targets = torch.full_like(logits, label_smoothing / (num_classes - 1))
                    smoothed_targets.scatter_(1, batch_targets.unsqueeze(1), 1.0 - label_smoothing)
                    loss = torch.sum(-smoothed_targets * torch.log_softmax(logits, dim=1), dim=1).mean()
                loss.backward()
                self.optimizer.step()
                epoch_running_loss += loss.item() * batch_features.size(0)
            scheduler.step()
            if verbose:
                logger.info(f"MLP epoch {epoch+1}/{epochs} loss: {epoch_running_loss / len(tensor_dataset):.4f}")
        return self

    def predict_proba(self, raw_features):
        self.model.eval()
        with torch.no_grad():
            feature_tensor = self._prepare_inputs(raw_features)
            logits = self.model(feature_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, raw_features):
        return self.predict_proba(raw_features).argmax(axis=1)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self


def cross_validate_models(feature_matrix, raw_target_labels, n_splits=5,
                           lightgbm_params=None, xgboost_params=None,
                           include_mlp=True):
    encoded_target_labels = encode_labels(raw_target_labels)
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lightgbm_oof_probabilities = np.zeros((len(feature_matrix), 3))
    xgboost_oof_probabilities = np.zeros((len(feature_matrix), 3))
    catboost_oof_probabilities = np.zeros((len(feature_matrix), 3))
    mlp_oof_probabilities = np.zeros((len(feature_matrix), 3))
    lightgbm_models, xgboost_models, catboost_models, mlp_models = [], [], [], []

    logger.info(f"Starting {n_splits}-Fold CV | Train size: {len(feature_matrix):,}")

    for fold_index, (train_indices, val_indices) in enumerate(stratified_kfold.split(feature_matrix, encoded_target_labels)):
        logger.info(f"{'='*55}")
        logger.info(f"FOLD {fold_index+1}/{n_splits}")
        logger.info(f"{'='*55}")

        train_features, val_features = feature_matrix.iloc[train_indices], feature_matrix.iloc[val_indices]
        train_targets, val_targets = encoded_target_labels[train_indices], encoded_target_labels[val_indices]

        logger.info(f"Train: {len(train_features):,} | Val: {len(val_features):,}")

        logger.info("Training LightGBM...")
        lightgbm_model = train_lightgbm(train_features, train_targets, val_features, val_targets, lightgbm_params)
        lightgbm_oof_probabilities[val_indices] = lightgbm_model.predict(val_features)
        lightgbm_balanced_accuracy = balanced_accuracy_score(val_targets, lightgbm_oof_probabilities[val_indices].argmax(1))
        logger.info(f"LightGBM BA: {lightgbm_balanced_accuracy:.4f}")
        lightgbm_models.append(LightGBMEstimatorWrapper(lightgbm_model))

        logger.info("Training XGBoost...")
        xgboost_model = train_xgboost(train_features, train_targets, val_features, val_targets, xgboost_params)
        xgboost_oof_probabilities[val_indices] = xgboost_model.predict_proba(val_features)
        xgboost_balanced_accuracy = balanced_accuracy_score(val_targets, xgboost_oof_probabilities[val_indices].argmax(1))
        logger.info(f"XGBoost BA: {xgboost_balanced_accuracy:.4f}")
        xgboost_models.append(xgboost_model)

        logger.info("Training CatBoost...")
        catboost_model = train_catboost(train_features, train_targets, val_features, val_targets)
        catboost_oof_probabilities[val_indices] = catboost_model.predict_proba(val_features)
        catboost_balanced_accuracy = balanced_accuracy_score(val_targets, catboost_oof_probabilities[val_indices].argmax(1))
        logger.info(f"CatBoost BA: {catboost_balanced_accuracy:.4f}")
        catboost_models.append(catboost_model)

        if include_mlp:
            logger.info("Training Tabular MLP...")
            mlp_model = TabularMLPClassifier(input_dim=train_features.shape[1], n_classes=3, hidden_dims=(128, 64), dropout=0.2)
            mlp_model.fit(train_features, train_targets, epochs=12, batch_size=256, verbose=False, label_smoothing=0.05, use_focal_loss=True)
            mlp_oof_probabilities[val_indices] = mlp_model.predict_proba(val_features)
            mlp_balanced_accuracy = balanced_accuracy_score(val_targets, mlp_oof_probabilities[val_indices].argmax(1))
            logger.info(f"MLP BA: {mlp_balanced_accuracy:.4f}")
            mlp_models.append(mlp_model)

        if include_mlp:
            fold_ensemble_probabilities = (lightgbm_oof_probabilities[val_indices]*0.55 +
                        xgboost_oof_probabilities[val_indices]*0.25 +
                        catboost_oof_probabilities[val_indices]*0.10 +
                        mlp_oof_probabilities[val_indices]*0.10)
        else:
            fold_ensemble_probabilities = (lightgbm_oof_probabilities[val_indices]*0.60 +
                        xgboost_oof_probabilities[val_indices]*0.35 +
                        catboost_oof_probabilities[val_indices]*0.05)
        fold_ensemble_balanced_accuracy = balanced_accuracy_score(val_targets, fold_ensemble_probabilities.argmax(1))
        logger.info(f"Fold {fold_index+1} Ensemble BA: {fold_ensemble_balanced_accuracy:.4f}")

    if include_mlp:
        ensemble_oof_probabilities = (lightgbm_oof_probabilities*0.55 + xgboost_oof_probabilities*0.25 + catboost_oof_probabilities*0.10 + mlp_oof_probabilities*0.10)
    else:
        ensemble_oof_probabilities = (lightgbm_oof_probabilities*0.60 + xgboost_oof_probabilities*0.35 + catboost_oof_probabilities*0.05)
    lightgbm_final_score = balanced_accuracy_score(encoded_target_labels, lightgbm_oof_probabilities.argmax(1))
    xgboost_final_score = balanced_accuracy_score(encoded_target_labels, xgboost_oof_probabilities.argmax(1))
    catboost_final_score = balanced_accuracy_score(encoded_target_labels, catboost_oof_probabilities.argmax(1))
    ensemble_final_score = balanced_accuracy_score(encoded_target_labels, ensemble_oof_probabilities.argmax(1))

    logger.info(f"{'='*55}")
    logger.info(f"FINAL CV RESULTS")
    logger.info(f"LightGBM  OOF: {lightgbm_final_score:.4f}")
    logger.info(f"XGBoost   OOF: {xgboost_final_score:.4f}")
    logger.info(f"CatBoost  OOF: {catboost_final_score:.4f}")
    if include_mlp:
        mlp_final_score = balanced_accuracy_score(encoded_target_labels, mlp_oof_probabilities.argmax(1))
        logger.info(f"MLP       OOF: {mlp_final_score:.4f}")
    logger.info(f"Ensemble  OOF: {ensemble_final_score:.4f}")
    logger.info(f"{'='*55}")

    return lightgbm_models, xgboost_models, catboost_models, mlp_models, lightgbm_oof_probabilities, xgboost_oof_probabilities, catboost_oof_probabilities, mlp_oof_probabilities


class MetaLearner:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(max_iter=2000, random_state=random_state)

    @staticmethod
    def _prepare_features(base_probabilities):
        probability_array = np.asarray(base_probabilities, dtype=float)
        if probability_array.ndim == 3:
            return probability_array.reshape(probability_array.shape[0], -1)
        if probability_array.ndim == 2:
            return probability_array
        raise ValueError(f'Expected 2D or 3D probabilities, got shape {probability_array.shape}')

    def fit(self, base_probabilities, y):
        meta_features = self._prepare_features(base_probabilities)
        self.model.fit(meta_features, y)
        return self

    def predict_proba(self, base_probabilities):
        meta_features = self._prepare_features(base_probabilities)
        return self.model.predict_proba(meta_features)

    def predict(self, base_probabilities):
        meta_features = self._prepare_features(base_probabilities)
        return self.model.predict(meta_features)


def train_meta_learner(base_probabilities, y, random_state=42):
    return MetaLearner(random_state=random_state).fit(base_probabilities, y)


def save_models(lightgbm_models, xgboost_models, catboost_models, output_dir='models', mlp_models=None):
    os.makedirs(output_dir, exist_ok=True)
    for fold_index, model in enumerate(lightgbm_models):
        model.save_model(f'{output_dir}/lgb_fold{fold_index+1}.txt')
    for fold_index, model in enumerate(xgboost_models):
        model.save_model(f'{output_dir}/xgb_fold{fold_index+1}.json')
    for fold_index, model in enumerate(catboost_models):
        model.save_model(f'{output_dir}/cat_fold{fold_index+1}.cbm')
    if mlp_models is not None:
        for fold_index, model in enumerate(mlp_models):
            model.save(f'{output_dir}/mlp_fold{fold_index+1}.pt')
    logger.info(f"Saved {len(lightgbm_models)} LGB + "
                f"{len(xgboost_models)} XGB + "
                f"{len(catboost_models)} CAT models to '{output_dir}'")


def run_adversarial_validation(train_features: pd.DataFrame, test_features: pd.DataFrame,
                                n_splits: int = 5, threshold: float = 0.70) -> dict:
    from sklearn.metrics import roc_auc_score

    logger.info(f"Running Adversarial Validation | Train: {len(train_features):,} | Test: {len(test_features):,}")
    logger.info(f"Drift Threshold: {threshold:.4f}")

    combined_features = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
    adversarial_labels = np.concatenate([
        np.zeros(len(train_features), dtype=int),
        np.ones(len(test_features), dtype=int)
    ])

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(combined_features))
    feature_importance_sum = np.zeros(train_features.shape[1])

    for fold_index, (train_indices, val_indices) in enumerate(stratified_kfold.split(combined_features, adversarial_labels)):
        train_fold_features, val_fold_features = combined_features.iloc[train_indices], combined_features.iloc[val_indices]
        train_fold_labels, val_fold_labels = adversarial_labels[train_indices], adversarial_labels[val_indices]

        train_dataset = lgb.Dataset(train_fold_features, label=train_fold_labels)
        val_dataset = lgb.Dataset(val_fold_features, label=val_fold_labels, reference=train_dataset)

        lightgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_jobs': -1,
            'seed': 42,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'min_child_samples': 20,
        }

        callbacks = [lgb.early_stopping(30, verbose=False),
              lgb.log_evaluation(period=-1)]

        trained_model = lgb.train(lightgbm_params, train_dataset, num_boost_round=500,
                         valid_sets=[val_dataset], callbacks=callbacks)

        oof_predictions[val_indices] = trained_model.predict(val_fold_features)
        feature_importance_sum += trained_model.feature_importance(importance_type='gain')

        fold_roc_auc = roc_auc_score(val_fold_labels, oof_predictions[val_indices])
        logger.info(f"Fold {fold_index+1}/{n_splits} Adversarial AUC: {fold_roc_auc:.4f}")

    overall_roc_auc = roc_auc_score(adversarial_labels, oof_predictions)
    avg_feature_importance = feature_importance_sum / n_splits

    feature_importance_dataframe = pd.DataFrame({
        'feature': train_features.columns,
        'drift_score': avg_feature_importance.astype(float),
    }).sort_values('drift_score', ascending=False).reset_index(drop=True)

    is_drifted = bool(overall_roc_auc > threshold)

    logger.info(f"Adversarial Validation OOF ROC-AUC: {overall_roc_auc:.4f}")
    logger.info(f"Distribution Shift Detected: {is_drifted} (threshold: {threshold:.4f})")
    logger.info(f"\nTop 10 Drifted Features:")
    logger.info(feature_importance_dataframe.head(10).to_string(index=False))

    return {
        'roc_auc': float(overall_roc_auc),
        'feature_importances': feature_importance_dataframe,
        'is_drifted': is_drifted,
    }


def train_stacking_meta_model(lightgbm_oof_probabilities: np.ndarray, xgboost_oof_probabilities: np.ndarray,
                               catboost_oof_probabilities: np.ndarray, train_targets,
                               n_splits: int = 5) -> dict:
    from sklearn.linear_model import LogisticRegression

    logger.info(f"Training Stacking Meta-Model | Train size: {len(train_targets):,}")

    if isinstance(train_targets, pd.Series):
        encoded_train_targets = encode_labels(train_targets)
    else:
        encoded_train_targets = train_targets

    meta_features = np.concatenate([lightgbm_oof_probabilities, xgboost_oof_probabilities, catboost_oof_probabilities], axis=1)
    logger.info(f"Meta-feature shape: {meta_features.shape} (base models: LGB + XGB + CAT)")

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_balanced_accuracy_scores = []

    for fold_index, (train_indices, val_indices) in enumerate(stratified_kfold.split(meta_features, encoded_train_targets)):
        train_meta_features, val_meta_features = meta_features[train_indices], meta_features[val_indices]
        train_fold_targets, val_fold_targets = encoded_train_targets[train_indices], encoded_train_targets[val_indices]

        fold_meta_model = LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            random_state=42,
        )
        fold_meta_model.fit(train_meta_features, train_fold_targets)

        val_predictions = fold_meta_model.predict(val_meta_features)
        fold_balanced_accuracy = balanced_accuracy_score(val_fold_targets, val_predictions)
        cv_balanced_accuracy_scores.append(fold_balanced_accuracy)

        logger.info(f"Fold {fold_index+1}/{n_splits} Meta-Model BA: {fold_balanced_accuracy:.4f}")

    cv_balanced_accuracy_scores = np.array(cv_balanced_accuracy_scores)

    final_meta_model = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
    )
    final_meta_model.fit(meta_features, encoded_train_targets)

    logger.info(f"{'='*55}")
    logger.info(f"Meta-Model CV Results:")
    logger.info(f"Mean BA: {cv_balanced_accuracy_scores.mean():.4f} ± {cv_balanced_accuracy_scores.std():.4f}")
    logger.info(f"{'='*55}")

    return {
        'meta_model': final_meta_model,
        'cv_scores': cv_balanced_accuracy_scores,
        'mean_cv_score': cv_balanced_accuracy_scores.mean(),
        'std_cv_score': cv_balanced_accuracy_scores.std(),
    }


def save_meta_model(meta_model, output_path: str):
    import joblib
    joblib.dump(meta_model, output_path)
    logger.info(f"Meta-model saved to: {output_path}")
