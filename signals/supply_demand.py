"""
Supply & Demand Zone Detection
Identifies institutional accumulation/distribution zones based on price action patterns.

Algorithm:
1. Find consolidation zones (low volatility, tight range)
2. Detect explosive moves away from consolidation (supply/demand imbalance)
3. Score zone strength (volume, touch count, time since formation)
4. Calculate proximity of current price to nearest zones
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Type of supply/demand zone."""
    DEMAND = "DEMAND"  # Accumulation zone (bullish)
    SUPPLY = "SUPPLY"  # Distribution zone (bearish)


@dataclass
class Zone:
    """Represents a supply or demand zone."""
    zone_type: ZoneType
    base_low: float
    base_high: float
    base_start_idx: int
    base_end_idx: int
    rally_drop_pct: float  # Magnitude of move away from base
    volume_ratio: float    # Volume during rally/drop vs base volume
    touch_count: int = 0   # Number of times zone has been tested
    last_touch_idx: Optional[int] = None
    is_fresh: bool = True  # Fresh zones (untested) are stronger

    @property
    def zone_midpoint(self) -> float:
        """Calculate zone midpoint."""
        return (self.base_low + self.base_high) / 2

    @property
    def zone_width(self) -> float:
        """Calculate zone width."""
        return self.base_high - self.base_low

    @property
    def strength_score(self) -> float:
        """
        Calculate zone strength (0-1).

        Factors:
        - Rally/drop magnitude (stronger move = stronger zone)
        - Volume ratio (higher volume = institutional activity)
        - Freshness (untested zones are more reliable)
        - Touch count (tested zones weaken over time)
        """
        # Base score from magnitude
        magnitude_score = min(1.0, abs(self.rally_drop_pct) / 0.20)  # 20% move = max score

        # Volume score (2x volume = max score)
        volume_score = min(1.0, self.volume_ratio / 2.0)

        # Freshness penalty
        freshness_score = 1.0 if self.is_fresh else 0.5

        # Touch penalty (each touch reduces strength by 20%)
        touch_penalty = max(0.2, 1.0 - (self.touch_count * 0.2))

        # Combined score
        return (magnitude_score * 0.4 + volume_score * 0.3) * freshness_score * touch_penalty


class SupplyDemandDetector:
    """
    Detects supply and demand zones from price action.

    Supply Zone (Distribution):
    - Price consolidates (low volatility)
    - Sharp drop occurs (sellers overwhelm buyers)
    - Zone = consolidation area before drop

    Demand Zone (Accumulation):
    - Price consolidates (low volatility)
    - Sharp rally occurs (buyers overwhelm sellers)
    - Zone = consolidation area before rally
    """

    def __init__(
        self,
        base_min_candles: int = 3,
        base_max_candles: int = 15,
        base_volatility_threshold: float = 0.02,  # 2% max range for consolidation
        rally_drop_threshold: float = 0.05,        # 5% minimum move to qualify
        volume_threshold: float = 1.2,             # 1.2x base volume during move
        proximity_threshold: float = 0.03          # 3% proximity to consider "near zone"
    ):
        """
        Initialize supply/demand detector.

        Args:
            base_min_candles: Minimum candles for consolidation base
            base_max_candles: Maximum candles for consolidation base
            base_volatility_threshold: Max price range for consolidation (%)
            rally_drop_threshold: Minimum % move to create zone
            volume_threshold: Minimum volume ratio (move vs base)
            proximity_threshold: Distance threshold for "near zone" (%)
        """
        self.base_min_candles = base_min_candles
        self.base_max_candles = base_max_candles
        self.base_volatility_threshold = base_volatility_threshold
        self.rally_drop_threshold = rally_drop_threshold
        self.volume_threshold = volume_threshold
        self.proximity_threshold = proximity_threshold

    def detect_zones(self, df: pd.DataFrame, lookback: int = 100) -> List[Zone]:
        """
        Detect all supply and demand zones in the given price data.

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of recent candles to analyze

        Returns:
            List of Zone objects
        """
        if len(df) < lookback:
            lookback = len(df)

        df_recent = df.iloc[-lookback:].copy()
        zones = []

        # Scan for consolidation bases followed by explosive moves
        for i in range(len(df_recent) - self.base_max_candles - 5):
            # Try different base lengths
            for base_length in range(self.base_min_candles, self.base_max_candles + 1):
                if i + base_length + 5 >= len(df_recent):
                    break

                base_candles = df_recent.iloc[i:i + base_length]

                # Check if this is a valid consolidation base
                if self._is_consolidation(base_candles):
                    # Check for rally (demand zone)
                    rally_zone = self._detect_rally(df_recent, i, base_length)
                    if rally_zone:
                        zones.append(rally_zone)

                    # Check for drop (supply zone)
                    drop_zone = self._detect_drop(df_recent, i, base_length)
                    if drop_zone:
                        zones.append(drop_zone)

        # Remove overlapping zones (keep stronger ones)
        zones = self._filter_overlapping_zones(zones)

        # Update zone touch counts
        zones = self._update_touch_counts(zones, df_recent)

        return zones

    def _is_consolidation(self, candles: pd.DataFrame) -> bool:
        """
        Check if candles represent a consolidation (tight range, low volatility).

        Args:
            candles: DataFrame of OHLC candles

        Returns:
            True if consolidation, False otherwise
        """
        if len(candles) < self.base_min_candles:
            return False

        high = candles['high'].max()
        low = candles['low'].min()
        close = candles['close'].iloc[-1]

        # Check range (should be tight)
        range_pct = (high - low) / close if close > 0 else 0

        return range_pct <= self.base_volatility_threshold

    def _detect_rally(self, df: pd.DataFrame, base_start: int, base_length: int) -> Optional[Zone]:
        """
        Detect demand zone (consolidation followed by rally).

        Args:
            df: Full DataFrame
            base_start: Index where base starts
            base_length: Length of consolidation base

        Returns:
            Zone object if valid rally detected, None otherwise
        """
        base_end = base_start + base_length
        if base_end + 5 >= len(df):
            return None

        base_candles = df.iloc[base_start:base_end]
        rally_candles = df.iloc[base_end:base_end + 5]

        base_high = base_candles['high'].max()
        base_low = base_candles['low'].min()
        base_close = base_candles['close'].iloc[-1]
        base_volume = base_candles['volume'].mean()

        rally_high = rally_candles['high'].max()
        rally_volume = rally_candles['volume'].mean()

        # Calculate rally magnitude
        rally_pct = (rally_high - base_close) / base_close if base_close > 0 else 0

        # Check if rally is strong enough
        if rally_pct < self.rally_drop_threshold:
            return None

        # Check volume confirmation
        volume_ratio = rally_volume / base_volume if base_volume > 0 else 0
        if volume_ratio < self.volume_threshold:
            return None

        return Zone(
            zone_type=ZoneType.DEMAND,
            base_low=base_low,
            base_high=base_high,
            base_start_idx=base_start,
            base_end_idx=base_end - 1,
            rally_drop_pct=rally_pct,
            volume_ratio=volume_ratio,
            is_fresh=True
        )

    def _detect_drop(self, df: pd.DataFrame, base_start: int, base_length: int) -> Optional[Zone]:
        """
        Detect supply zone (consolidation followed by drop).

        Args:
            df: Full DataFrame
            base_start: Index where base starts
            base_length: Length of consolidation base

        Returns:
            Zone object if valid drop detected, None otherwise
        """
        base_end = base_start + base_length
        if base_end + 5 >= len(df):
            return None

        base_candles = df.iloc[base_start:base_end]
        drop_candles = df.iloc[base_end:base_end + 5]

        base_high = base_candles['high'].max()
        base_low = base_candles['low'].min()
        base_close = base_candles['close'].iloc[-1]
        base_volume = base_candles['volume'].mean()

        drop_low = drop_candles['low'].min()
        drop_volume = drop_candles['volume'].mean()

        # Calculate drop magnitude
        drop_pct = (drop_low - base_close) / base_close if base_close > 0 else 0

        # Check if drop is strong enough
        if drop_pct > -self.rally_drop_threshold:
            return None

        # Check volume confirmation
        volume_ratio = drop_volume / base_volume if base_volume > 0 else 0
        if volume_ratio < self.volume_threshold:
            return None

        return Zone(
            zone_type=ZoneType.SUPPLY,
            base_low=base_low,
            base_high=base_high,
            base_start_idx=base_start,
            base_end_idx=base_end - 1,
            rally_drop_pct=drop_pct,
            volume_ratio=volume_ratio,
            is_fresh=True
        )

    def _filter_overlapping_zones(self, zones: List[Zone]) -> List[Zone]:
        """
        Remove overlapping zones, keeping stronger ones.

        Args:
            zones: List of Zone objects

        Returns:
            Filtered list of zones
        """
        if not zones:
            return []

        # Sort by strength (descending)
        zones_sorted = sorted(zones, key=lambda z: z.strength_score, reverse=True)

        filtered = []
        for zone in zones_sorted:
            # Check if overlaps with any existing filtered zone
            overlaps = False
            for existing in filtered:
                if self._zones_overlap(zone, existing):
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(zone)

        return filtered

    def _zones_overlap(self, zone1: Zone, zone2: Zone) -> bool:
        """Check if two zones overlap."""
        return not (zone1.base_high < zone2.base_low or zone2.base_high < zone1.base_low)

    def _update_touch_counts(self, zones: List[Zone], df: pd.DataFrame) -> List[Zone]:
        """
        Update touch counts for each zone based on price action after formation.

        Args:
            zones: List of Zone objects
            df: Price DataFrame

        Returns:
            Updated zones with touch counts
        """
        for zone in zones:
            touch_count = 0
            last_touch_idx = None

            # Look at candles after zone formation
            for i in range(zone.base_end_idx + 6, len(df)):
                candle = df.iloc[i]

                # Check if price touched the zone
                if zone.base_low <= candle['low'] <= zone.base_high or \
                   zone.base_low <= candle['high'] <= zone.base_high:
                    touch_count += 1
                    last_touch_idx = i
                    zone.is_fresh = False

            zone.touch_count = touch_count
            zone.last_touch_idx = last_touch_idx

        return zones

    def get_nearest_zones(
        self,
        zones: List[Zone],
        current_price: float
    ) -> Tuple[Optional[Zone], Optional[Zone]]:
        """
        Get nearest demand and supply zones to current price.

        Args:
            zones: List of Zone objects
            current_price: Current price

        Returns:
            Tuple of (nearest_demand_zone, nearest_supply_zone)
        """
        demand_zones = [z for z in zones if z.zone_type == ZoneType.DEMAND and z.zone_midpoint < current_price]
        supply_zones = [z for z in zones if z.zone_type == ZoneType.SUPPLY and z.zone_midpoint > current_price]

        nearest_demand = None
        if demand_zones:
            nearest_demand = max(demand_zones, key=lambda z: z.zone_midpoint)

        nearest_supply = None
        if supply_zones:
            nearest_supply = min(supply_zones, key=lambda z: z.zone_midpoint)

        return nearest_demand, nearest_supply

    def calculate_zone_score(
        self,
        zones: List[Zone],
        current_price: float
    ) -> float:
        """
        Calculate supply/demand voting score (-1 to +1).

        Score interpretation:
        - +1: Strong demand zone nearby (bullish)
        - -1: Strong supply zone nearby (bearish)
        - 0: Neutral (no zones nearby or balanced)

        Args:
            zones: List of Zone objects
            current_price: Current price

        Returns:
            Score between -1 and +1
        """
        nearest_demand, nearest_supply = self.get_nearest_zones(zones, current_price)

        demand_score = 0.0
        supply_score = 0.0

        # Calculate demand contribution
        if nearest_demand:
            distance_pct = abs(current_price - nearest_demand.zone_midpoint) / current_price
            proximity_score = max(0, 1 - (distance_pct / self.proximity_threshold))
            demand_score = proximity_score * nearest_demand.strength_score

        # Calculate supply contribution
        if nearest_supply:
            distance_pct = abs(current_price - nearest_supply.zone_midpoint) / current_price
            proximity_score = max(0, 1 - (distance_pct / self.proximity_threshold))
            supply_score = proximity_score * nearest_supply.strength_score

        # Net score (demand is positive, supply is negative)
        return demand_score - supply_score

    def get_zone_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract supply/demand features for ML model.

        Args:
            df: Price DataFrame

        Returns:
            Dictionary of features
        """
        zones = self.detect_zones(df)
        current_price = df['close'].iloc[-1]

        nearest_demand, nearest_supply = self.get_nearest_zones(zones, current_price)

        features = {
            'sd_score': self.calculate_zone_score(zones, current_price),
            'sd_demand_distance': 0.0,
            'sd_demand_strength': 0.0,
            'sd_supply_distance': 0.0,
            'sd_supply_strength': 0.0,
            'sd_demand_fresh': 0.0,
            'sd_supply_fresh': 0.0,
            'sd_zone_count': len(zones),
            'sd_net_strength': 0.0
        }

        if nearest_demand:
            features['sd_demand_distance'] = abs(current_price - nearest_demand.zone_midpoint) / current_price
            features['sd_demand_strength'] = nearest_demand.strength_score
            features['sd_demand_fresh'] = 1.0 if nearest_demand.is_fresh else 0.0

        if nearest_supply:
            features['sd_supply_distance'] = abs(current_price - nearest_supply.zone_midpoint) / current_price
            features['sd_supply_strength'] = nearest_supply.strength_score
            features['sd_supply_fresh'] = 1.0 if nearest_supply.is_fresh else 0.0

        # Net strength (demand - supply)
        features['sd_net_strength'] = features['sd_demand_strength'] - features['sd_supply_strength']

        return features
