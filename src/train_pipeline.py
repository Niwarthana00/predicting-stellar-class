import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from src.data.preprocessing import preprocess
from src.features.engineering import build_features, FEATURE_COLS
from src.models.train import (
    cross_validate_models,
    save_models,
    tune_lightgbm,
    tune_xgboost,
    encode_labels,
    REVERSE_MAP,
    pseudo_label_with_model,
    compute_permutation_importance,
    explain_with_shap,
    train_meta_learner,
)


def build_dataset(data_dir: str, use_processed: bool = True):
    from src.data.preprocessing import load_data

    processed_train_path = os.path.join(data_dir, 'X_train.csv')
    processed_y_path = os.path.join(data_dir, 'y_train.csv')
    processed_test_path = os.path.join(data_dir, 'X_test.csv')

    if use_processed and os.path.exists(processed_train_path) and os.path.exists(processed_y_path) and os.path.exists(processed_test_path):
        processed_X = pd.read_csv(processed_train_path)
        processed_y = pd.read_csv(processed_y_path).squeeze()
        processed_test = pd.read_csv(processed_test_path)
        if set(FEATURE_COLS).issubset(processed_X.columns) and set(FEATURE_COLS).issubset(processed_test.columns):
            X = processed_X[FEATURE_COLS].copy()
            y = processed_y.copy()
            X_test = processed_test[FEATURE_COLS].copy()
            print('Using processed feature files from disk.')
            return X, y, X_test

    raw_data_dir = os.path.join(os.path.dirname(data_dir), 'raw')
    if not os.path.exists(raw_data_dir):
        raw_data_dir = data_dir

    train_raw, test_raw = load_data(
        os.path.join(raw_data_dir, 'train.csv'),
        os.path.join(raw_data_dir, 'test.csv'),
    )
    train_prep = preprocess(train_raw)
    test_prep = preprocess(test_raw)
    train_feat = build_features(train_prep)
    test_feat = build_features(test_prep)

    X = train_feat[FEATURE_COLS].copy()
    y = train_feat['class'].copy()
    X_test = test_feat[FEATURE_COLS].copy()

    return X, y, X_test


def save_metrics(metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)


def build_feature_importance_table(models, X: pd.DataFrame) -> pd.DataFrame:
    feature_names = list(X.columns)
    importance_values = np.zeros(len(feature_names), dtype=float)

    for model in models:
        raw_importance = np.asarray(model.feature_importance(importance_type='gain'), dtype=float)
        if len(raw_importance) != len(feature_names):
            if len(raw_importance) > len(feature_names):
                raw_importance = raw_importance[:len(feature_names)]
            else:
                raw_importance = np.pad(raw_importance, (0, len(feature_names) - len(raw_importance)), mode='constant')
        importance_values += raw_importance

    importance_values /= max(len(models), 1)
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
    }).sort_values('importance', ascending=False).reset_index(drop=True)


def save_feature_importance(importance_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(y_true, y_pred, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=['GALAXY', 'QSO', 'STAR'])
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['GALAXY', 'QSO', 'STAR'],
                yticklabels=['GALAXY', 'QSO', 'STAR'])
    axes[0].set_title('Confusion Matrix (Counts)')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                xticklabels=['GALAXY', 'QSO', 'STAR'],
                yticklabels=['GALAXY', 'QSO', 'STAR'])
    axes[1].set_title('Confusion Matrix (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


class LightGBMBoosterWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.booster.predict(X).argmax(axis=1)

    def predict_proba(self, X):
        return self.booster.predict(X)


def save_permutation_importance(model, X, y, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    X_sample = X.iloc[:5000] if hasattr(X, 'iloc') else X[:5000]
    y_sample = y.iloc[:5000] if hasattr(y, 'iloc') else y[:5000]
    wrapped_model = LightGBMBoosterWrapper(model)
    perm_imp = compute_permutation_importance(wrapped_model, X_sample, y_sample, n_repeats=10)
    perm_imp.to_csv(os.path.join(output_dir, 'permutation_importance.csv'), index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=perm_imp.head(15), x='importance_mean', y='feature', hue='feature', dodge=False, legend=False, palette='rocket')
    plt.title('Permutation Importance', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_predictions(preds, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'prediction': preds}).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Train stellar classification ensemble and save artifacts')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--use-pseudo', action='store_true')
    parser.add_argument('--pseudo-threshold', type=float, default=0.95)
    parser.add_argument('--n-trials-lgb', type=int, default=20)
    parser.add_argument('--n-trials-xgb', type=int, default=20)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--no-mlp', action='store_true', help='Disable the tabular neural network branch')
    parser.add_argument('--use-meta-learner', action='store_true', help='Train a small meta-learner on base-model probabilities')
    parser.add_argument('--tune-weights', action='store_true', help='Try a small grid of ensemble weights for the base models')
    args = parser.parse_args()

    X, y, X_test = build_dataset(args.data_dir, use_processed=True)
    print('Loaded data:', X.shape, y.shape, X_test.shape)

    if args.use_pseudo:
        from lightgbm import LGBMClassifier
        seed_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=127,
            objective='multiclass',
            n_jobs=-1,
            random_state=42,
        )
        seed_model.fit(X, encode_labels(y))
        X_train, y_train, _ = pseudo_label_with_model(
            X, y, X_test, seed_model, confidence_threshold=args.pseudo_threshold
        )
        print(f'Pseudo-labeled samples added: {len(y_train) - len(y)}')
    else:
        X_train, y_train = X.copy(), y.copy()

    y_enc = encode_labels(y_train)
    lgb_best_params = tune_lightgbm(X_train, y_enc, n_trials=args.n_trials_lgb)
    xgb_best_params = tune_xgboost(X_train, y_enc, n_trials=args.n_trials_xgb)

    lgb_models, xgb_models, cat_models, mlp_models, oof_lgb, oof_xgb, oof_cat, oof_mlp = cross_validate_models(
        X_train, y_train, n_splits=args.n_splits, lgb_params=lgb_best_params, xgb_params=xgb_best_params,
        include_mlp=not args.no_mlp
    )

    save_models(lgb_models, xgb_models, cat_models, output_dir=args.model_dir, mlp_models=mlp_models)

    if args.tune_weights:
        if mlp_models is not None:
            candidates = [
                (0.55, 0.25, 0.10, 0.10),
                (0.60, 0.25, 0.10, 0.05),
                (0.50, 0.30, 0.10, 0.10),
                (0.65, 0.20, 0.10, 0.05),
            ]
            scores = []
            for weights in candidates:
                w_lgb, w_xgb, w_cat, w_mlp = weights
                blended = (oof_lgb * w_lgb + oof_xgb * w_xgb + oof_cat * w_cat + oof_mlp * w_mlp)
                score = balanced_accuracy_score(y_enc, blended.argmax(axis=1))
                scores.append((score, weights))
            best_score, best_weights = max(scores, key=lambda item: item[0])
            oof_ensemble = (oof_lgb * best_weights[0] + oof_xgb * best_weights[1] + oof_cat * best_weights[2] + oof_mlp * best_weights[3])
            print('Best ensemble weights:', best_weights, 'score:', round(best_score, 4))
        else:
            candidates = [(0.6, 0.3, 0.1), (0.7, 0.2, 0.1), (0.5, 0.4, 0.1)]
            scores = []
            for weights in candidates:
                blended = (oof_lgb * weights[0] + oof_xgb * weights[1] + oof_cat * weights[2])
                score = balanced_accuracy_score(y_enc, blended.argmax(axis=1))
                scores.append((score, weights))
            best_score, best_weights = max(scores, key=lambda item: item[0])
            oof_ensemble = (oof_lgb * best_weights[0] + oof_xgb * best_weights[1] + oof_cat * best_weights[2])
            print('Best ensemble weights:', best_weights, 'score:', round(best_score, 4))
    elif mlp_models is not None:
        oof_ensemble = (oof_lgb*0.55 + oof_xgb*0.25 + oof_cat*0.10 + oof_mlp*0.10)
    else:
        oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3

    if args.use_meta_learner:
        base_probs = np.stack([oof_lgb, oof_xgb, oof_cat] + ([oof_mlp] if mlp_models is not None else []), axis=2)
        meta_model = train_meta_learner(base_probs, y_enc)
        meta_probs = meta_model.predict_proba(base_probs)
        oof_ensemble = meta_probs

    oof_pred = oof_ensemble.argmax(axis=1)
    oof_pred_labels = [REVERSE_MAP[i] for i in oof_pred]

    metrics = {
        'balanced_accuracy': round(float(balanced_accuracy_score(y_enc, oof_pred)), 4),
        'classification_report': classification_report(y_train, oof_pred_labels, target_names=['GALAXY', 'QSO', 'STAR'], output_dict=True),
    }
    save_metrics(metrics, args.output_dir)

    importance_df = build_feature_importance_table(lgb_models, X_train)
    save_feature_importance(importance_df, args.output_dir)

    save_confusion_matrix(y_train, oof_pred_labels, args.output_dir)
    save_permutation_importance(lgb_models[0], X_train, y_enc, args.output_dir)
    save_predictions(oof_pred_labels, args.output_dir)

    print('Training complete.')
    print('Artifacts saved to:', args.output_dir)
    print('Models saved to:', args.model_dir)


if __name__ == '__main__':
    main()
