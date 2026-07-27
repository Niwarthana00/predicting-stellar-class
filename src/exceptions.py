"""
Custom exceptions for the stellar classification pipeline.

This module defines domain-specific exceptions for handling data quality,
distribution drift, and model-related errors in production ML pipelines.
"""


class DataDriftException(Exception):
    """
    Raised when adversarial validation detects significant distribution shift.
    
    This exception indicates that the training and test data distributions
    are too different, which may compromise model generalization and inference.
    
    Attributes:
        roc_auc: The adversarial validation ROC-AUC score (0.0 - 1.0)
        threshold: The AUC threshold that was exceeded
        message: Descriptive error message with context
    """
    
    def __init__(self, roc_auc: float, threshold: float, 
                 feature_importance_df=None, message: str = None):
        """
        Initialize DataDriftException.
        
        Args:
            roc_auc: The adversarial validation ROC-AUC score
            threshold: The threshold that was exceeded
            feature_importance_df: DataFrame with drifted features (optional)
            message: Custom error message (optional)
        """
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
    """Raised when feature engineering fails or produces invalid outputs."""
    pass


class ModelTrainingException(Exception):
    """Raised when model training encounters critical errors."""
    pass
