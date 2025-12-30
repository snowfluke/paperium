"""
Ensemble Voting System for Ultra-High Precision

Combines multiple models + technical analysis for consensus-based trading:
1. XGBoost (Primary Model)
2. LightGBM (Fast gradient boosting)
3. CatBoost (Category-optimized boosting)
4. Supply/Demand Zones (Technical analysis vote)

Voting Strategy:
- Require 3/4 models to agree for signal
- Use weighted voting based on recent performance
- S/D zones act as tiebreaker
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import pickle

import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available, ensemble will use reduced models")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available, ensemble will use reduced models")

from signals.supply_demand import SupplyDemandDetector

logger = logging.getLogger(__name__)


class EnsembleVoter:
    """
    Multi-model ensemble with voting mechanism.

    Each model gets one vote. Final signal requires minimum votes (default 3/4).
    """

    def __init__(
        self,
        min_votes: int = 3,
        xgb_weight: float = 1.0,
        lgb_weight: float = 1.0,
        cat_weight: float = 1.0,
        sd_weight: float = 0.8,  # S/D zones weighted slightly lower
        sd_threshold: float = 0.3  # S/D score threshold for positive vote
    ):
        """
        Initialize ensemble voter.

        Args:
            min_votes: Minimum votes required for signal (3/4 recommended)
            xgb_weight: Weight for XGBoost vote
            lgb_weight: Weight for LightGBM vote
            cat_weight: Weight for CatBoost vote
            sd_weight: Weight for S/D zone vote
            sd_threshold: Minimum S/D score for positive vote
        """
        self.min_votes = min_votes
        self.weights = {
            'xgb': xgb_weight,
            'lgb': lgb_weight if LIGHTGBM_AVAILABLE else 0.0,
            'cat': cat_weight if CATBOOST_AVAILABLE else 0.0,
            'sd': sd_weight
        }
        self.sd_threshold = sd_threshold

        # Models
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.lgb_model = None
        self.cat_model = None
        self.sd_detector = SupplyDemandDetector()

        # Performance tracking for adaptive weights
        self.model_performance = {
            'xgb': {'correct': 0, 'total': 0},
            'lgb': {'correct': 0, 'total': 0},
            'cat': {'correct': 0, 'total': 0},
            'sd': {'correct': 0, 'total': 0}
        }

    def train_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Train all ensemble models.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_estimators: Number of trees
            max_depth: Max tree depth
            learning_rate: Learning rate

        Returns:
            Training metrics for each model
        """
        metrics = {}

        # 1. Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            gamma=0.01,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X, y)
        xgb_pred = self.xgb_model.predict(X)
        metrics['xgb_accuracy'] = (xgb_pred == y).mean()

        # 2. Train LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=0.01,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            self.lgb_model.fit(X, y)
            lgb_pred = self.lgb_model.predict(X)
            metrics['lgb_accuracy'] = (lgb_pred == y).mean()

        # 3. Train CatBoost (if available)
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost...")
            self.cat_model = cb.CatBoostClassifier(
                iterations=n_estimators,
                depth=max_depth,
                learning_rate=learning_rate,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            )
            self.cat_model.fit(X, y)
            cat_pred = self.cat_model.predict(X).flatten()
            metrics['cat_accuracy'] = (cat_pred == y).mean()

        logger.info(f"Ensemble trained: {metrics}")
        return metrics

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        df: pd.DataFrame  # Original price df for S/D zones
    ) -> Tuple[pd.Series, pd.Series, Dict[str, pd.Series]]:
        """
        Get ensemble predictions with voting.

        Args:
            X: Feature DataFrame
            df: Original price DataFrame (for S/D detection)

        Returns:
            Tuple of (final_predictions, ensemble_confidence, individual_votes)
        """
        if self.xgb_model is None:
            raise ValueError("Ensemble not trained. Call train_ensemble() first.")

        votes = {}
        probas = {}

        # 1. XGBoost vote
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        votes['xgb'] = (xgb_proba > 0.5).astype(int)
        probas['xgb'] = xgb_proba

        # 2. LightGBM vote
        if self.lgb_model is not None:
            lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
            votes['lgb'] = (lgb_proba > 0.5).astype(int)
            probas['lgb'] = lgb_proba
        else:
            votes['lgb'] = np.zeros(len(X), dtype=int)
            probas['lgb'] = np.zeros(len(X))

        # 3. CatBoost vote
        if self.cat_model is not None:
            cat_proba = self.cat_model.predict_proba(X)[:, 1]
            votes['cat'] = (cat_proba > 0.5).astype(int)
            probas['cat'] = cat_proba
        else:
            votes['cat'] = np.zeros(len(X), dtype=int)
            probas['cat'] = np.zeros(len(X))

        # 4. S/D Zone vote (only for latest/current price)
        sd_scores = []
        for i in range(len(df)):
            # Get historical data up to this point
            df_slice = df.iloc[:i+1]
            sd_features = self.sd_detector.get_zone_features(df_slice)
            sd_score = sd_features['sd_score']
            sd_scores.append(sd_score)

        sd_scores = np.array(sd_scores)
        votes['sd'] = (sd_scores > self.sd_threshold).astype(int)
        probas['sd'] = (sd_scores + 1) / 2  # Convert -1 to 1 range → 0 to 1

        # Calculate weighted votes
        weighted_votes = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for model_name, vote in votes.items():
            weighted_votes += vote * self.weights[model_name]

        # Normalize to 0-1 range
        weighted_votes /= total_weight

        # Final prediction: requires minimum vote threshold
        # For 4 models with min_votes=3: need 75% agreement (3/4)
        vote_threshold = self.min_votes / len(votes)
        final_predictions = (weighted_votes >= vote_threshold).astype(int)

        # Ensemble confidence (average of all probabilities)
        ensemble_confidence = np.mean([probas[m] for m in probas.keys()], axis=0)

        # Convert to Series
        final_predictions = pd.Series(final_predictions, index=X.index)
        ensemble_confidence = pd.Series(ensemble_confidence, index=X.index)
        individual_votes = {k: pd.Series(v, index=X.index) for k, v in votes.items()}

        return final_predictions, ensemble_confidence, individual_votes

    def update_performance(self, model_name: str, was_correct: bool):
        """
        Update model performance tracking for adaptive weighting.

        Args:
            model_name: Name of model ('xgb', 'lgb', 'cat', 'sd')
            was_correct: Whether prediction was correct
        """
        if model_name in self.model_performance:
            self.model_performance[model_name]['total'] += 1
            if was_correct:
                self.model_performance[model_name]['correct'] += 1

    def get_model_accuracies(self) -> Dict[str, float]:
        """Get recent accuracy for each model."""
        accuracies = {}
        for model_name, perf in self.model_performance.items():
            if perf['total'] > 0:
                accuracies[model_name] = perf['correct'] / perf['total']
            else:
                accuracies[model_name] = 0.5  # Neutral default

        return accuracies

    def adaptive_reweight(self, decay: float = 0.9):
        """
        Adjust model weights based on recent performance.

        Better performing models get higher weight.

        Args:
            decay: Decay factor for older performance (0.9 = 10% decay)
        """
        accuracies = self.get_model_accuracies()

        # Normalize accuracies to weights
        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            for model_name in self.weights:
                if model_name in accuracies:
                    # Update weight with decay
                    new_weight = accuracies[model_name] / total_accuracy * 4  # Scale to ~1.0
                    self.weights[model_name] = (
                        decay * self.weights[model_name] +
                        (1 - decay) * new_weight
                    )

        logger.debug(f"Adaptive weights updated: {self.weights}")

    def save(self, path_prefix: str):
        """
        Save all ensemble models.

        Args:
            path_prefix: Prefix for model files (e.g., 'models/ensemble_')
        """
        # Save XGBoost
        if self.xgb_model:
            self.xgb_model.save_model(f"{path_prefix}xgb.json")

        # Save LightGBM
        if self.lgb_model:
            with open(f"{path_prefix}lgb.pkl", 'wb') as f:
                pickle.dump(self.lgb_model, f)

        # Save CatBoost
        if self.cat_model:
            self.cat_model.save_model(f"{path_prefix}cat.cbm")

        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'performance': self.model_performance,
            'min_votes': self.min_votes,
            'sd_threshold': self.sd_threshold
        }
        with open(f"{path_prefix}metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Ensemble saved to {path_prefix}*")

    def load(self, path_prefix: str):
        """
        Load all ensemble models.

        Args:
            path_prefix: Prefix for model files
        """
        # Load XGBoost
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(f"{path_prefix}xgb.json")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost: {e}")

        # Load LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                with open(f"{path_prefix}lgb.pkl", 'rb') as f:
                    self.lgb_model = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LightGBM: {e}")

        # Load CatBoost
        if CATBOOST_AVAILABLE:
            try:
                self.cat_model = cb.CatBoostClassifier()
                self.cat_model.load_model(f"{path_prefix}cat.cbm")
            except Exception as e:
                logger.warning(f"Failed to load CatBoost: {e}")

        # Load metadata
        try:
            with open(f"{path_prefix}metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                self.weights = metadata.get('weights', self.weights)
                self.model_performance = metadata.get('performance', self.model_performance)
                self.min_votes = metadata.get('min_votes', self.min_votes)
                self.sd_threshold = metadata.get('sd_threshold', self.sd_threshold)
        except Exception as e:
            logger.warning(f"Failed to load ensemble metadata: {e}")

        logger.info(f"Ensemble loaded from {path_prefix}*")
