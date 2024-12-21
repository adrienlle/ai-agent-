import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm

class MarketRegime(Enum):
    TREND_STRONG_UP = "TREND_STRONG_UP"
    TREND_UP = "TREND_UP"
    SIDEWAYS = "SIDEWAYS"
    TREND_DOWN = "TREND_DOWN"
    TREND_STRONG_DOWN = "TREND_STRONG_DOWN"
    VOLATILE_UP = "VOLATILE_UP"
    VOLATILE_DOWN = "VOLATILE_DOWN"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"

@dataclass
class MarketCondition:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    support_resistance: List[float]
    volume_profile: Dict[str, float]
    liquidity_score: float
    market_impact: float

class MarketRegimeDetector:
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.hmm = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour le HMM."""
        features = []
        
        # Rendements et volatilité
        returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)
        
        # Indicateurs techniques
        rsi = talib.RSI(data['close'].values)
        macd, signal, _ = talib.MACD(data['close'].values)
        
        # Volume features
        volume_ma = data['volume'].rolling(window=20).mean().fillna(0)
        volume_std = data['volume'].rolling(window=20).std().fillna(0)
        
        features = np.column_stack([
            returns,
            volatility,
            rsi,
            macd,
            signal,
            volume_ma,
            volume_std
        ])
        
        # Normalisation
        features = self.scaler.fit_transform(features)
        
        # Réduction dimensionnelle
        features = self.pca.fit_transform(features)
        
        return features
    
    def fit(self, data: pd.DataFrame):
        """Entraîne le HMM sur les données historiques."""
        features = self.prepare_features(data)
        self.hmm.fit(features)
    
    def predict_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Prédit le régime de marché actuel."""
        features = self.prepare_features(data)
        state = self.hmm.predict(features)[-1]
        
        # Analyse des caractéristiques du dernier état
        last_features = features[-20:]  # Dernières 20 observations
        
        # Calcul des statistiques
        returns_mean = np.mean(last_features[:, 0])
        volatility = np.std(last_features[:, 0])
        trend = np.polyfit(range(len(last_features)), last_features[:, 0], 1)[0]
        
        # Détermination du régime
        if trend > 0.01 and volatility < 0.02:
            return MarketRegime.TREND_STRONG_UP
        elif trend > 0.005:
            return MarketRegime.TREND_UP
        elif trend < -0.01 and volatility < 0.02:
            return MarketRegime.TREND_STRONG_DOWN
        elif trend < -0.005:
            return MarketRegime.TREND_DOWN
        elif volatility > 0.03:
            return MarketRegime.VOLATILE_UP if returns_mean > 0 else MarketRegime.VOLATILE_DOWN
        else:
            return MarketRegime.SIDEWAYS

class VolumeProfileAnalyzer:
    def __init__(self, price_levels: int = 50):
        self.price_levels = price_levels
        
    def analyze(self, 
                data: pd.DataFrame,
                window: int = 1000) -> Dict[str, Union[np.ndarray, float]]:
        """Analyse le profil de volume."""
        recent_data = data.tail(window)
        
        # Création des niveaux de prix
        price_range = np.linspace(
            recent_data['low'].min(),
            recent_data['high'].max(),
            self.price_levels
        )
        
        # Calcul du Volume Profile
        volume_profile = np.zeros(self.price_levels)
        for i in range(len(recent_data)):
            idx_range = np.where(
                (price_range >= recent_data['low'].iloc[i]) &
                (price_range <= recent_data['high'].iloc[i])
            )[0]
            volume_profile[idx_range] += recent_data['volume'].iloc[i] / len(idx_range)
        
        # Point of Control (POC)
        poc_idx = np.argmax(volume_profile)
        poc_price = price_range[poc_idx]
        
        # Value Area
        total_volume = np.sum(volume_profile)
        value_area_volume = 0
        value_area_indices = [poc_idx]
        i_above, i_below = poc_idx + 1, poc_idx - 1
        
        while value_area_volume < 0.68 * total_volume and \
              (i_above < len(volume_profile) or i_below >= 0):
            vol_above = volume_profile[i_above] if i_above < len(volume_profile) else 0
            vol_below = volume_profile[i_below] if i_below >= 0 else 0
            
            if vol_above > vol_below:
                value_area_indices.append(i_above)
                value_area_volume += vol_above
                i_above += 1
            else:
                value_area_indices.append(i_below)
                value_area_volume += vol_below
                i_below -= 1
        
        value_area_high = price_range[max(value_area_indices)]
        value_area_low = price_range[min(value_area_indices)]
        
        return {
            'volume_profile': volume_profile,
            'price_levels': price_range,
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'value_area_volume_ratio': value_area_volume / total_volume
        }

class LiquidityAnalyzer:
    def __init__(self, depth_levels: int = 20):
        self.depth_levels = depth_levels
        
    def analyze_orderbook(self, 
                         orderbook: Dict[str, List[List[float]]],
                         current_price: float) -> Dict[str, float]:
        """Analyse la liquidité dans l'orderbook."""
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        
        # Calcul de la profondeur de marché
        bid_depth = np.sum(bids[:self.depth_levels, 1])
        ask_depth = np.sum(asks[:self.depth_levels, 1])
        
        # Calcul du spread
        spread = (asks[0][0] - bids[0][0]) / current_price
        
        # Calcul de l'impact de marché
        def calculate_impact(orders: np.ndarray, size: float) -> float:
            cumsum = np.cumsum(orders[:, 1])
            idx = np.searchsorted(cumsum, size)
            if idx >= len(orders):
                return float('inf')
            return abs(orders[idx][0] - orders[0][0]) / orders[0][0]
        
        impact_100k = calculate_impact(asks, 100000) if len(asks) > 0 else float('inf')
        impact_1m = calculate_impact(asks, 1000000) if len(asks) > 0 else float('inf')
        
        # Score de liquidité composite
        liquidity_score = (
            0.4 * (1 / (1 + spread)) +
            0.3 * (bid_depth / (bid_depth + ask_depth)) +
            0.3 * (1 / (1 + impact_100k))
        )
        
        return {
            'spread': spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'impact_100k': impact_100k,
            'impact_1m': impact_1m,
            'liquidity_score': liquidity_score
        }

class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
    
    def analyze(self, 
                data: pd.DataFrame,
                orderbook: Optional[Dict] = None) -> MarketCondition:
        """Analyse complète des conditions de marché."""
        # Détection du régime
        regime = self.regime_detector.predict_regime(data)
        
        # Analyse du volume
        volume_profile = self.volume_analyzer.analyze(data)
        
        # Calcul de la volatilité
        returns = np.log(data['close'] / data['close'].shift(1))
        volatility = returns.rolling(window=20).std().iloc[-1]
        
        # Force de la tendance
        adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values)
        trend_strength = adx[-1] / 100.0
        
        # Support et résistance
        pivots = self._calculate_pivots(data)
        
        # Analyse de la liquidité
        if orderbook:
            liquidity = self.liquidity_analyzer.analyze_orderbook(
                orderbook,
                data['close'].iloc[-1]
            )
            liquidity_score = liquidity['liquidity_score']
            market_impact = liquidity['impact_100k']
        else:
            liquidity_score = 0.5  # Valeur par défaut
            market_impact = 0.01
        
        return MarketCondition(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            support_resistance=pivots,
            volume_profile=volume_profile,
            liquidity_score=liquidity_score,
            market_impact=market_impact
        )
    
    def _calculate_pivots(self, data: pd.DataFrame, window: int = 20) -> List[float]:
        """Calcule les niveaux de support et résistance."""
        pivots = []
        
        # Calcul des pivots classiques
        pp = (data['high'] + data['low'] + data['close']) / 3
        r1 = 2 * pp - data['low']
        s1 = 2 * pp - data['high']
        r2 = pp + (data['high'] - data['low'])
        s2 = pp - (data['high'] - data['low'])
        
        # Ajout des niveaux de Fibonacci
        range_high_low = data['high'].max() - data['low'].min()
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in fib_levels:
            fib_level = data['low'].min() + range_high_low * level
            pivots.append(fib_level)
        
        # Ajout des pivots classiques
        pivots.extend([
            pp.iloc[-1],
            r1.iloc[-1],
            s1.iloc[-1],
            r2.iloc[-1],
            s2.iloc[-1]
        ])
        
        # Suppression des doublons et tri
        pivots = sorted(list(set(pivots)))
        
        return pivots

class AdvancedStrategyExecutor:
    def __init__(self):
        self.analyzer = AdvancedTechnicalAnalyzer()
        
    def generate_trade_plan(self,
                           market_condition: MarketCondition,
                           risk_params: Dict[str, float]) -> Dict:
        """Génère un plan de trading basé sur l'analyse technique avancée."""
        
        # Ajustement de la taille de position selon les conditions
        position_size = self._calculate_position_size(
            market_condition,
            risk_params['base_position_size'],
            risk_params['max_position_size']
        )
        
        # Génération des niveaux d'entrée
        entry_levels = self._generate_entry_levels(
            market_condition,
            position_size
        )
        
        # Génération des niveaux de sortie
        exit_levels = self._generate_exit_levels(
            market_condition,
            entry_levels['primary_entry']
        )
        
        return {
            'position_size': position_size,
            'entry_levels': entry_levels,
            'exit_levels': exit_levels,
            'trade_type': self._determine_trade_type(market_condition),
            'execution_strategy': self._generate_execution_strategy(market_condition)
        }
    
    def _calculate_position_size(self,
                               condition: MarketCondition,
                               base_size: float,
                               max_size: float) -> float:
        """Calcule la taille optimale de la position."""
        # Facteurs d'ajustement
        liquidity_factor = np.clip(condition.liquidity_score, 0.1, 1.0)
        volatility_factor = np.clip(1 - condition.volatility * 5, 0.1, 1.0)
        trend_factor = np.clip(condition.trend_strength, 0.1, 1.0)
        
        # Score composite
        score = (liquidity_factor * 0.4 +
                volatility_factor * 0.3 +
                trend_factor * 0.3)
        
        # Taille finale
        position_size = base_size * score
        return min(position_size, max_size)
    
    def _generate_entry_levels(self,
                             condition: MarketCondition,
                             position_size: float) -> Dict:
        """Génère des niveaux d'entrée optimisés."""
        
        # Sélection des niveaux pertinents
        support_resistance = condition.support_resistance
        volume_profile = condition.volume_profile
        
        # Niveau d'entrée principal (POC)
        primary_entry = volume_profile['poc']
        
        # Niveaux d'entrée secondaires
        secondary_entries = []
        for level in support_resistance:
            if abs(level - primary_entry) / primary_entry < 0.02:  # Dans une marge de 2%
                secondary_entries.append(level)
        
        # Distribution de la taille de position
        if len(secondary_entries) > 0:
            primary_size = position_size * 0.6
            secondary_sizes = [(position_size * 0.4) / len(secondary_entries)] * len(secondary_entries)
        else:
            primary_size = position_size
            secondary_sizes = []
        
        return {
            'primary_entry': primary_entry,
            'primary_size': primary_size,
            'secondary_entries': list(zip(secondary_entries, secondary_sizes))
        }
    
    def _generate_exit_levels(self,
                            condition: MarketCondition,
                            entry_price: float) -> Dict:
        """Génère des niveaux de sortie optimisés."""
        
        # Calcul des take-profits basé sur la volatilité et le régime
        volatility_multiplier = 2.0 if condition.volatility < 0.05 else 1.5
        
        take_profits = [
            entry_price * (1 + volatility_multiplier * condition.volatility),
            entry_price * (1 + volatility_multiplier * 2 * condition.volatility),
            entry_price * (1 + volatility_multiplier * 3 * condition.volatility)
        ]
        
        # Stop loss dynamique
        stop_loss = entry_price * (1 - condition.volatility * 3)
        
        # Trailing stop
        trailing_stop = {
            'activation_price': take_profits[0],
            'callback_rate': min(0.01 + condition.volatility, 0.03)
        }
        
        return {
            'take_profits': take_profits,
            'stop_loss': stop_loss,
            'trailing_stop': trailing_stop
        }
    
    def _determine_trade_type(self, condition: MarketCondition) -> str:
        """Détermine le type de trade optimal."""
        if condition.regime in [MarketRegime.TREND_STRONG_UP, MarketRegime.TREND_UP]:
            return 'TREND_FOLLOWING_LONG'
        elif condition.regime in [MarketRegime.TREND_STRONG_DOWN, MarketRegime.TREND_DOWN]:
            return 'TREND_FOLLOWING_SHORT'
        elif condition.regime == MarketRegime.SIDEWAYS:
            return 'RANGE_TRADING'
        elif condition.regime in [MarketRegime.VOLATILE_UP, MarketRegime.VOLATILE_DOWN]:
            return 'MOMENTUM_SCALPING'
        else:
            return 'BREAKOUT_TRADING'
    
    def _generate_execution_strategy(self, condition: MarketCondition) -> Dict:
        """Génère une stratégie d'exécution optimisée."""
        
        # Paramètres de base selon la liquidité
        base_params = {
            'order_type': 'LIMIT' if condition.liquidity_score > 0.7 else 'MARKET',
            'time_in_force': 'GTC',
            'post_only': condition.liquidity_score > 0.8
        }
        
        # Paramètres d'exécution avancés
        execution_params = {
            'slippage_tolerance': max(0.001, condition.market_impact * 0.5),
            'execution_style': self._determine_execution_style(condition),
            'order_book_depth': int(20 * condition.liquidity_score),
            'min_execution_size': 100,
            'max_execution_size': 10000 * condition.liquidity_score
        }
        
        return {**base_params, **execution_params}
    
    def _determine_execution_style(self, condition: MarketCondition) -> str:
        """Détermine le style d'exécution optimal."""
        if condition.liquidity_score > 0.8 and condition.volatility < 0.02:
            return 'PASSIVE'
        elif condition.liquidity_score < 0.3 or condition.volatility > 0.05:
            return 'AGGRESSIVE'
        else:
            return 'BALANCED'
