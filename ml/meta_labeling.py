"""
Meta-Labeling for Trade Quality Filtering

Meta-labeling is a machine learning technique where a secondary model predicts
whether the primary model's signal will be profitable. This dramatically improves
precision by filtering out low-quality signals.

Process:
1. Primary model generates signals (buy/sell)
2. Meta-label model predicts "Will this signal succeed?" (1/0)
3. Only execute trades where both models agree

This increases win rate while maintaining reasonable trade frequency.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class MetaLabeler:
    """
    Secondary model that predicts trade quality.

    Features used:
    - Primary model confidence
    - Market regime indicators
    - Recent performance metrics
    - Technical setup quality scores
    - S/D zone alignment
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,  # Min meta-label confidence to trade
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1
    ):
        """
        Initialize meta-labeler.

        Args:
            confidence_threshold: Minimum confidence to approve trade
            n_estimators: XGBoost trees for meta-model
            max_depth: Max depth of meta-model trees (keep shallow to avoid overfitting)
            learning_rate: Learning rate for meta-model
        """
        self.confidence_threshold = confidence_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list = []

    def create_meta_features(
        self,
        df: pd.DataFrame,
        primary_signals: pd.Series,
        primary_confidence: pd.Series
    ) -> pd.DataFrame:
        """
        Create features for meta-labeling model.

        Args:
            df: DataFrame with OHLCV and technical indicators
            primary_signals: Primary model signals (1/0)
            primary_confidence: Primary model confidence scores (0-1)

        Returns:
            DataFrame with meta-features
        """
        meta_df = pd.DataFrame(index=df.index)

        # 1. Primary model features
        meta_df['primary_confidence'] = primary_confidence
        meta_df['primary_signal'] = primary_signals

        # 2. Confidence-based features
        meta_df['confidence_zscore'] = (
            (primary_confidence - primary_confidence.rolling(20).mean()) /
            primary_confidence.rolling(20).std().replace(0, 1)
        )
        meta_df['confidence_percentile'] = primary_confidence.rolling(50).apply(
            lambda x: (x.iloc[-1] > x).mean()
        )

        # 3. Technical setup quality
        # RSI strength
        if 'rsi' in df.columns:
            meta_df['rsi_quality'] = np.where(
                (df['rsi'] >= 50) & (df['rsi'] <= 70),
                1.0,
                0.5
            )
        else:
            meta_df['rsi_quality'] = 0.5

        # Trend alignment
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            meta_df['trend_alignment'] = (
                (df['close'] > df['sma_20']).astype(int) +
                (df['sma_20'] > df['sma_50']).astype(int)
            ) / 2.0
        else:
            meta_df['trend_alignment'] = 0.5

        # Volume confirmation
        if 'volume' in df.columns and 'volume_sma' in df.columns:
            meta_df['volume_confirmation'] = (
                df['volume'] / df['volume_sma'].replace(0, 1)
            ).clip(0, 3) / 3.0  # Normalize to 0-1
        else:
            meta_df['volume_confirmation'] = 0.5

        # 4. Volatility features
        if 'atr' in df.columns:
            atr_pct = df['atr'] / df['close']
            meta_df['volatility_regime'] = (
                atr_pct.rolling(20).mean() / atr_pct.rolling(60).mean()
            ).replace([np.inf, -np.inf], 1.0)
        else:
            meta_df['volatility_regime'] = 1.0

        # 5. Supply/Demand alignment (if available)
        if 'sd_score' in df.columns:
            meta_df['sd_alignment'] = df['sd_score']
            meta_df['sd_demand_strength'] = df.get('sd_demand_strength', 0)
        else:
            meta_df['sd_alignment'] = 0.0
            meta_df['sd_demand_strength'] = 0.0

        # 6. Market structure
        if 'gap' in df.columns and 'gap_followthrough' in df.columns:
            meta_df['gap_quality'] = df['gap'].abs() * df['gap_followthrough']
        else:
            meta_df['gap_quality'] = 0.0

        # 7. Momentum quality
        if 'return_5d' in df.columns and 'return_20d' in df.columns:
            meta_df['momentum_consistency'] = np.sign(df['return_5d']) == np.sign(df['return_20d'])
            meta_df['momentum_consistency'] = meta_df['momentum_consistency'].astype(float)
        else:
            meta_df['momentum_consistency'] = 0.5

        # 8. Recent performance proxy (consecutive wins/losses)
        if 'target_direction' in df.columns:
            meta_df['recent_accuracy'] = df['target_direction'].rolling(10).mean()
        else:
            meta_df['recent_accuracy'] = 0.5

        # Fill NaN values
        meta_df = meta_df.fillna(0.5)

        self.feature_names = [
            'primary_confidence', 'confidence_zscore', 'confidence_percentile',
            'rsi_quality', 'trend_alignment', 'volume_confirmation',
            'volatility_regime', 'sd_alignment', 'sd_demand_strength',
            'gap_quality', 'momentum_consistency', 'recent_accuracy'
        ]

        return meta_df[self.feature_names]

    def create_meta_labels(
        self,
        df: pd.DataFrame,
        primary_signals: pd.Series,
        forward_returns: pd.Series,
        sl_threshold: float = -0.05,
        tp_threshold: float = 0.08
    ) -> pd.Series:
        """
        Create meta-labels: Did the primary signal succeed?

        A signal succeeds if:
        - Forward return > TP threshold (profit target hit)
        OR
        - Forward return > 0 AND forward return > SL threshold (profitable exit)

        Args:
            df: Price DataFrame
            primary_signals: Primary model signals (1/0)
            forward_returns: Forward returns (e.g., 5-day)
            sl_threshold: Stop loss threshold (negative)
            tp_threshold: Take profit threshold (positive)

        Returns:
            Series of meta-labels (1 = success, 0 = failure)
        """
        # Only label where primary model gave a signal
        meta_labels = pd.Series(0, index=df.index)

        signal_mask = primary_signals == 1

        # Success criteria
        success = (
            (forward_returns >= tp_threshold) |  # TP hit
            ((forward_returns > 0) & (forward_returns > sl_threshold))  # Profitable
        )

        meta_labels[signal_mask] = success[signal_mask].astype(int)

        return meta_labels

    def train(
        self,
        df: pd.DataFrame,
        primary_signals: pd.Series,
        primary_confidence: pd.Series,
        forward_returns: pd.Series,
        validate: bool = True
    ) -> dict:
        """
        Train the meta-labeling model.

        Args:
            df: DataFrame with OHLCV and indicators
            primary_signals: Primary model signals
            primary_confidence: Primary model confidence
            forward_returns: Forward returns for labeling
            validate: Whether to perform cross-validation

        Returns:
            Training metrics
        """
        # Create meta-features
        X_meta = self.create_meta_features(df, primary_signals, primary_confidence)

        # Create meta-labels
        y_meta = self.create_meta_labels(df, primary_signals, forward_returns)

        # Filter: only train on samples where primary model gave a signal
        signal_mask = primary_signals == 1
        X_train = X_meta[signal_mask]
        y_train = y_meta[signal_mask]

        if len(X_train) < 50:
            logger.warning(f"Insufficient meta-labeling data: {len(X_train)} samples")
            return {'status': 'insufficient_data'}

        # Train XGBoost meta-model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        metrics = {
            'train_samples': len(X_train),
            'positive_rate': y_train.mean()
        }

        # Validation
        if validate:
            val_metrics = self._validate(X_train, y_train)
            metrics.update(val_metrics)

        logger.info(f"Meta-labeler trained: {metrics}")

        return metrics

    def _validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 3
    ) -> dict:
        """
        Perform time-series cross-validation on meta-model.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits

        Returns:
            Validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        precisions = []
        recalls = []
        f1_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42
            )
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)

            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

        return {
            'cv_precision': float(np.mean(precisions)),
            'cv_recall': float(np.mean(recalls)),
            'cv_f1': float(np.mean(f1_scores))
        }

    def predict(
        self,
        df: pd.DataFrame,
        primary_signals: pd.Series,
        primary_confidence: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict which primary signals will succeed.

        Args:
            df: DataFrame with OHLCV and indicators
            primary_signals: Primary model signals
            primary_confidence: Primary model confidence

        Returns:
            Tuple of (filtered_signals, meta_confidence)
        """
        if self.model is None:
            logger.warning("Meta-model not trained, returning original signals")
            return primary_signals, primary_confidence

        # Create meta-features
        X_meta = self.create_meta_features(df, primary_signals, primary_confidence)

        # Get meta-predictions
        meta_proba = self.model.predict_proba(X_meta)[:, 1]  # Probability of success
        meta_predictions = (meta_proba >= self.confidence_threshold).astype(int)

        # Filter primary signals
        filtered_signals = primary_signals * meta_predictions

        # Combine confidences (geometric mean for conservative estimate)
        combined_confidence = np.sqrt(primary_confidence * meta_proba)

        return pd.Series(filtered_signals, index=df.index), pd.Series(combined_confidence, index=df.index)

    def save(self, path: str):
        """Save meta-labeler model."""
        if self.model:
            self.model.save_model(path)
            logger.info(f"Meta-labeler saved to {path}")

    def load(self, path: str):
        """Load meta-labeler model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        logger.info(f"Meta-labeler loaded from {path}")
