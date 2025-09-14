"""
XGBoost Model for Bitcoin Price Prediction
Traditional machine learning approach for 3-class classification
"""

import xgboost as xgb

def build_xgboost_model(max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42):
    """
    Create XGBoost classifier for 3-class prediction
    
    Args:
        max_depth: Maximum depth of trees
        learning_rate: Learning rate for boosting
        n_estimators: Number of estimators
        random_state: Random state for reproducibility
    """
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        eval_metric='mlogloss',
        verbosity=0  # Suppress warnings
    )
    return model