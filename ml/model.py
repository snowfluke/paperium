"""
Machine Learning Model Module
XGBoost-based prediction with daily self-refinement
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import pickle
import os
from datetime import datetime
import logging

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb

from .features import FeatureEngineer

logger = logging.getLogger(__name__)

# Global flag to show GPU warning only once per session
_gpu_warning_shown = False


class TradingModel:
    """
    XGBoost-based trading model with rolling window training.
    Implements daily self-refinement (EOD model update).
    """
    
    def __init__(self, config=None):
        """
        Initialize trading model.
        
        Args:
            config: MLConfig object (optional)
        """
        if config:
            self.training_window = config.training_window
            self.min_training_samples = getattr(config, 'min_training_samples', 60)
            self.n_estimators = config.n_estimators
            self.max_depth = config.max_depth
            self.learning_rate = config.learning_rate
            self.min_child_weight = config.min_child_weight
            self.use_gpu = getattr(config, 'use_gpu', False)
        else:
            self.training_window = 252
            self.min_training_samples = 40
            self.n_estimators = 100
            self.max_depth = 5
            self.learning_rate = 0.1
            self.min_child_weight = 3
            self.use_gpu = False
        
        self.feature_engineer = FeatureEngineer(config)
        self.model = None
        self.feature_names = None
        self.last_trained = None
        self.performance_history = []
    
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with configured parameters."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': 0.8,              # Gen 4 value (was 0.7 - too restrictive!)
            'colsample_bytree': 0.8,       # Gen 4 value (was 0.7 - too restrictive!)
            # Removed reg_alpha (was 0.1 - killing weak features!)
            # Removed gamma (was 0.1 - preventing splits!)
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }


        if self.use_gpu:
            global _gpu_warning_shown
            import sys
            # Check for MPS stability - Fallback to CPU on Mac to avoid binary crash
            if sys.platform == "darwin":
                if not _gpu_warning_shown:
                    logger.warning("XGBoost MPS (Metal) acceleration can be unstable on some Mac environments. Using high-performance CPU ('hist') instead.")
                    _gpu_warning_shown = True
                params['tree_method'] = 'hist'
                params['device'] = 'cpu'
            else:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda'
                
        return xgb.XGBClassifier(**params)
    
    def train(
        self, 
        df: pd.DataFrame,
        validate: bool = True,
        base_model: Optional[xgb.XGBClassifier] = None
    ) -> Dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            validate: Whether to perform validation
            base_model: Existing model to use as starting point (warm start)
            
        Returns:
            Dictionary of training metrics
        """
        if base_model is not None:
            logger.info("Retraining existing champion (Warm Start)...")
            self.model = base_model
            # Ensure feature names match if possible, or we'll error out during fit
        else:
            logger.info("Training fresh ML model...")
        
        # Create features
        X, y, _ = self.feature_engineer.create_features(df)
        
        # Clean data (handle inf and nan)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        
        if len(X) < self.min_training_samples:
            logger.warning(f"Insufficient data: {len(X)} < {self.min_training_samples}")
            return {}
        
        # Use rolling window (adaptive)
        current_window = min(len(X), self.training_window)
        X_train = X.iloc[-current_window:]
        y_train = y.iloc[-current_window:]
        
        self.feature_names = X_train.columns.tolist()
        
        # Train model
        import time
        train_start = time.time()
        if base_model is None:
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
        else:
            # Incremental learning: pass booster to xgb_model
            # Note: Feature set must be identical
            if self.model is not None:
                self.model.fit(X_train, y_train, xgb_model=base_model.get_booster())
        
        train_duration = time.time() - train_start
        self.last_trained = datetime.now()
        
        metrics = {
            'train_samples': len(X_train),
            'features': len(self.feature_names),
            'train_time_sec': round(train_duration, 4)
        }
        
        # Validate if requested
        if validate:
            val_metrics = self._validate(X_train, y_train)
            metrics.update(val_metrics)
        
        logger.info(f"Model trained in {train_duration:.4f}s: {metrics}")
        
        return metrics
    
    def _validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        accuracies = []
        precisions = []
        
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self._create_model()
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            
            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
        
        return {
            'cv_accuracy': float(np.mean(accuracies)),
            'cv_accuracy_std': float(np.std(accuracies)),
            'cv_precision': float(np.mean(precisions))
        }
    
    def daily_update(
        self, 
        df: pd.DataFrame,
        new_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Daily model update (self-refinement).
        
        Uses rolling window: drops oldest day, adds newest day.
        This is the key to the model's ability to adapt to changing market conditions.
        
        Args:
            df: Full historical DataFrame
            new_data: Optional new data to append
            
        Returns:
            Updated training metrics
        """
        logger.info("Performing daily model update...")
        
        if new_data is not None:
            df = pd.concat([df, new_data], ignore_index=True)
        
        # Retrain with latest data
        metrics = self.train(df, validate=True)
        
        if metrics:
            self.performance_history.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                **metrics
            })
        
        return metrics
    
    def predict(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with OHLCV and indicator columns
            
        Returns:
            Tuple of (class predictions, probability predictions)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = self.feature_engineer.prepare_inference_features(df)

        # Align features with training features
        if self.feature_names is None:
            self.feature_names = []
        missing_features = set(self.feature_names) - set(X.columns)
        for f in missing_features:
            X[f] = 0
        
        X = X[self.feature_names]
        
        # Clean data for GPU backend
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Use DMatrix to avoid device mismatch warning (explicit data transfer)
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Predict
        y_proba = self.model.get_booster().predict(dmat)
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_pred, y_proba
    
    def predict_latest(
        self, 
        df: pd.DataFrame
    ) -> Tuple[int, float]:
        """
        Predict for the latest (most recent) data point.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            Tuple of (direction prediction, probability)
        """
        y_pred, y_proba = self.predict(df)
        return y_pred[-1], y_proba[-1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        
        return self.feature_engineer.get_feature_importance(
            self.model, 
            self.feature_names
        )
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: File path for model
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        # Save metadata separately
        model_data = {
            # 'model': self.model,  <-- Don't pickle the model wrapper anymore
            'feature_names': self.feature_names,
            'last_trained': self.last_trained,
            'performance_history': self.performance_history,
            'config': {
                'training_window': self.training_window,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        # Save XGBoost model natively (JSON) to avoid version mismatch warnings
        if self.model is not None:
            xgb_path = path.replace('.pkl', '.json')
            # Use the booster's save_model to avoid sklearn wrapper metadata issues
            self.model.get_booster().save_model(xgb_path)

        
        logger.info(f"Model saved to {path} (metadata) and {xgb_path} (booster)")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: File path for model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load metadata
        self.feature_names = model_data['feature_names']
        
        # Load model: Check for native JSON first, fallback to pickle 'model' key for backward compatibility
        xgb_path = path.replace('.pkl', '.json')
        if os.path.exists(xgb_path):
            self.model = self._create_model()
            self.model.load_model(xgb_path)
            logger.info("  -> Loaded native XGBoost JSON model")
        elif 'model' in model_data and model_data['model'] is not None:
            self.model = model_data['model']
            logger.info("  -> Loaded legacy pickled XGBoost model")
        self.last_trained = model_data.get('last_trained')
        self.performance_history = model_data.get('performance_history', [])
        
        # Try to get last_trained from champion_metadata.json if not in pkl
        if self.last_trained is None:
            metadata_path = os.path.join(os.path.dirname(path), "champion_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if 'xgboost' in metadata and 'date' in metadata['xgboost']:
                        self.last_trained = metadata['xgboost']['date']
                except Exception:
                    pass
        
        logger.info(f"Model loaded from {path}, last trained: {self.last_trained}")
