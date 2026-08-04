import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import os

CLASS_INDEX_TO_LABEL = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}


class LightGBMBoosterWrapper:

    def __init__(self, lightgbm_booster):
        self.lightgbm_booster = lightgbm_booster

    def predict_proba(self, features):
        return self.lightgbm_booster.predict(features)

    def predict(self, features):
        return self.predict_proba(features).argmax(axis=1)

    def feature_importance(self, **kwargs):
        return self.lightgbm_booster.feature_importance(**kwargs)


def wrap_lightgbm_booster(lightgbm_booster):
    return LightGBMBoosterWrapper(lightgbm_booster)


def load_lightgbm_models(models_directory='models'):
    loaded_models = []
    if not os.path.exists(models_directory):
        print(f"Not found: {models_directory}")
        return loaded_models
    for filename in sorted(os.listdir(models_directory)):
        file_path = os.path.join(models_directory, filename)
        if os.path.isfile(file_path) and filename.startswith('lgb_fold') and filename.endswith('.txt'):
            lightgbm_booster = lgb.Booster(model_file=file_path)

            loaded_models.append(wrap_lightgbm_booster(lightgbm_booster))
    print(f"Loaded {len(loaded_models)} LightGBM from '{models_directory}'")
    return loaded_models


def load_xgboost_models(models_directory='models'):
    loaded_models = []
    if not os.path.exists(models_directory):
        return loaded_models
    for filename in sorted(os.listdir(models_directory)):
        file_path = os.path.join(models_directory, filename)
        if os.path.isfile(file_path) and filename.startswith('xgb_fold') and filename.endswith('.json'):
            xgboost_model = xgb.XGBClassifier()
            xgboost_model.load_model(file_path)
            loaded_models.append(xgboost_model)
    print(f"Loaded {len(loaded_models)} XGBoost from '{models_directory}'")
    return loaded_models


def load_catboost_models(models_directory='models'):
    loaded_models = []
    if not os.path.exists(models_directory):
        return loaded_models
    for filename in sorted(os.listdir(models_directory)):
        file_path = os.path.join(models_directory, filename)
        if os.path.isfile(file_path) and filename.startswith('cat_fold') and filename.endswith('.cbm'):
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(file_path)
            loaded_models.append(catboost_model)
    print(f"Loaded {len(loaded_models)} CatBoost from '{models_directory}'")
    return loaded_models


def compute_ensemble_class_probabilities(features: pd.DataFrame,
                                          lightgbm_models, xgboost_models,
                                          catboost_models=None,
                                          lightgbm_weight=0.60, xgboost_weight=0.35, catboost_weight=0.05) -> np.ndarray:
    lightgbm_probabilities = np.zeros((len(features), 3))
    xgboost_probabilities = np.zeros((len(features), 3))

    for model in lightgbm_models:
        if isinstance(model, lgb.Booster):
            model = wrap_lightgbm_booster(model)
        lightgbm_probabilities += model.predict_proba(features)
    lightgbm_probabilities /= len(lightgbm_models)

    for model in xgboost_models:
        xgboost_probabilities += model.predict_proba(features)
    xgboost_probabilities /= len(xgboost_models)

    if catboost_models and len(catboost_models) > 0:
        catboost_probabilities = np.zeros((len(features), 3))
        for model in catboost_models:
            catboost_probabilities += model.predict_proba(features)
        catboost_probabilities /= len(catboost_models)
        ensemble_probabilities = (lightgbm_probabilities * lightgbm_weight +
                                   xgboost_probabilities * xgboost_weight +
                                   catboost_probabilities * catboost_weight)
    else:
        total_weight = lightgbm_weight + xgboost_weight
        ensemble_probabilities = (lightgbm_probabilities * (lightgbm_weight / total_weight) +
                                   xgboost_probabilities * (xgboost_weight / total_weight))

    return ensemble_probabilities


def predict_class_labels(features: pd.DataFrame,
                          lightgbm_models, xgboost_models,
                          catboost_models=None,
                          lightgbm_weight=0.60, xgboost_weight=0.35, catboost_weight=0.05) -> np.ndarray:
    ensemble_probabilities = compute_ensemble_class_probabilities(
        features, lightgbm_models, xgboost_models, catboost_models,
        lightgbm_weight, xgboost_weight, catboost_weight)
    return np.array([CLASS_INDEX_TO_LABEL[i] for i in ensemble_probabilities.argmax(axis=1)])


def build_submission_dataframe(test_ids, probabilities, output_path=None, include_index=False):
    predicted_labels = np.array([CLASS_INDEX_TO_LABEL[i] for i in probabilities.argmax(axis=1)])
    submission_dataframe = pd.DataFrame({'id': test_ids, 'class': predicted_labels})
    if output_path is not None:
        submission_dataframe.to_csv(output_path, index=include_index)
    return submission_dataframe


def predict_single_sample(feature_dict: dict,
                           lightgbm_models, xgboost_models,
                           catboost_models, feature_columns: list,
                           lightgbm_weight=0.60, xgboost_weight=0.35, catboost_weight=0.05) -> dict:
    from src.data.preprocessing import preprocess
    from src.features.engineering import build_features

    dataframe = pd.DataFrame([feature_dict])
    dataframe = preprocess(dataframe)
    dataframe = build_features(dataframe)
    features = dataframe[feature_columns]

    probabilities = compute_ensemble_class_probabilities(
        features, lightgbm_models, xgboost_models, catboost_models,
        lightgbm_weight, xgboost_weight, catboost_weight)[0]
    predicted_class = CLASS_INDEX_TO_LABEL[probabilities.argmax()]

    return {
        'predicted_class': predicted_class,
        'probabilities': {
            'GALAXY': round(float(probabilities[0]), 4),
            'QSO':    round(float(probabilities[1]), 4),
            'STAR':   round(float(probabilities[2]), 4),
        }
    }


def predict_with_test_time_augmentation(lightgbm_models, xgboost_models, catboost_models,
                                         features: pd.DataFrame,
                                         noise_standard_deviation: float = 0.01,
                                         augmentation_repeats: int = 5,
                                         lightgbm_weight: float = 0.60,
                                         xgboost_weight: float = 0.35,
                                         catboost_weight: float = 0.05,
                                         random_seed: int = 42) -> np.ndarray:
    np.random.seed(random_seed)

    numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    accumulated_probabilities = np.zeros((len(features), 3))

    original_probabilities = compute_ensemble_class_probabilities(
        features, lightgbm_models, xgboost_models, catboost_models,
        lightgbm_weight=lightgbm_weight, xgboost_weight=xgboost_weight, catboost_weight=catboost_weight)
    accumulated_probabilities += original_probabilities

    for repeat_index in range(augmentation_repeats):
        augmented_features = features.copy()

        if numerical_columns:
            augmented_features[numerical_columns] = augmented_features[numerical_columns].astype(float)
            gaussian_noise = np.random.normal(loc=0.0, scale=noise_standard_deviation,
                                               size=(len(augmented_features), len(numerical_columns)))
            augmented_features[numerical_columns] += gaussian_noise

        augmented_probabilities = compute_ensemble_class_probabilities(
            augmented_features, lightgbm_models, xgboost_models, catboost_models,
            lightgbm_weight=lightgbm_weight, xgboost_weight=xgboost_weight, catboost_weight=catboost_weight)
        accumulated_probabilities += augmented_probabilities

    averaged_probabilities = accumulated_probabilities / (1 + augmentation_repeats)

    return averaged_probabilities


def load_meta_model(model_path: str):
    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Meta-model not found at: {model_path}")

    meta_model = joblib.load(model_path)
    print(f"Meta-model loaded from: {model_path}")
    return meta_model


def predict_with_meta_model(meta_model,
                             lightgbm_oof_probabilities: np.ndarray,
                             xgboost_oof_probabilities: np.ndarray,
                             catboost_oof_probabilities: np.ndarray) -> np.ndarray:
    meta_features = np.concatenate(
        [lightgbm_oof_probabilities, xgboost_oof_probabilities, catboost_oof_probabilities], axis=1)

    meta_probabilities = meta_model.predict_proba(meta_features)

    return meta_probabilities
