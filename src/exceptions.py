class DataDriftException(Exception):
    
    def __init__(self, roc_auc: float, threshold: float, 
                 feature_importance_df=None, message: str = None):

        self.roc_auc = roc_auc
        self.threshold = threshold
        self.feature_importance_df = feature_importance_df
        
        if message is None:
            message = (
                f"Data drift detected: Adversarial Validation ROC-AUC {roc_auc:.4f} "
                f"exceeds threshold {threshold:.4f}. Train/test distributions are "
                f"significantly different, which may compromise model generalization."
            )
        
        super().__init__(message)


class FeatureEngineeringException(Exception):
    pass


class ModelTrainingException(Exception):
    pass
