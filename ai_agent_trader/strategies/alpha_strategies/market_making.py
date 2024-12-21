import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
import pandas as pd
from scipy.stats import norm
from collections import deque

class MarketMakingStyle(Enum):
    PASSIVE = "PASSIVE"
    NEUTRAL = "NEUTRAL"
    AGGRESSIVE = "AGGRESSIVE"
    ADAPTIVE = "ADAPTIVE"
    PREDATORY = "PREDATORY"

@dataclass
class MarketMakingState:
    inventory: float
    cash: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    inventory_risk: float
    spread_capture: float
    order_flow_imbalance: float
    toxic_flow_ratio: float
    effective_spread: float

class InventoryManager:
    def __init__(self, 
                 target_inventory: float = 0,
                 max_inventory: float = 100,
                 risk_aversion: float = 0.1):
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.risk_aversion = risk_aversion
        
        # Modèle de risque d'inventaire
        self.volatility_window = deque(maxlen=100)
        self.position_limits = self._calculate_position_limits()
    
    def _calculate_position_limits(self) -> Dict[str, float]:
        """Calcule les limites de position dynamiques."""
        vol = np.std(self.volatility_window) if self.volatility_window else 0.02
        
        # Limites adaptatives basées sur la volatilité
        dynamic_max = self.max_inventory * np.exp(-self.risk_aversion * vol)
        
        return {
            'soft_upper': 0.8 * dynamic_max,
            'soft_lower': -0.8 * dynamic_max,
            'hard_upper': dynamic_max,
            'hard_lower': -dynamic_max
        }
    
    def calculate_inventory_risk(self, 
                               current_inventory: float,
                               market_price: float,
                               volatility: float) -> float:
        """Calcule le risque d'inventaire."""
        # Mise à jour de la fenêtre de volatilité
        self.volatility_window.append(volatility)
        
        # Calcul du risque
        inventory_deviation = current_inventory - self.target_inventory
        position_value = inventory_deviation * market_price
        
        # VaR de l'inventaire
        var_95 = position_value * volatility * norm.ppf(0.95)
        
        # Score de risque normalisé
        risk_score = abs(var_95) / (self.max_inventory * market_price)
        
        return risk_score
    
    def get_inventory_adjustments(self,
                                current_inventory: float,
                                market_price: float,
                                volatility: float) -> Dict[str, float]:
        """Calcule les ajustements de prix basés sur l'inventaire."""
        risk_score = self.calculate_inventory_risk(
            current_inventory,
            market_price,
            volatility
        )
        
        # Skew des prix basé sur l'inventaire
        inventory_skew = (current_inventory - self.target_inventory) / self.max_inventory
        
        # Ajustements non-linéaires
        bid_adjustment = -np.sign(inventory_skew) * (risk_score ** 2)
        ask_adjustment = np.sign(inventory_skew) * (risk_score ** 2)
        
        return {
            'bid_adjustment': bid_adjustment,
            'ask_adjustment': ask_adjustment,
            'risk_score': risk_score
        }

class SpreadOptimizer:
    def __init__(self,
                 base_spread: float = 0.001,
                 min_spread: float = 0.0001,
                 max_spread: float = 0.01):
        self.base_spread = base_spread
        self.min_spread = min_spread
        self.max_spread = max_spread
        
        # Historique pour l'apprentissage
        self.spread_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        
        # Modèle d'optimisation
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Construit un modèle pour l'optimisation du spread."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse'
        )
        
        return model
    
    def optimize_spread(self,
                       market_state: Dict,
                       inventory_risk: float) -> float:
        """Optimise le spread basé sur les conditions de marché."""
        # Features pour le modèle
        features = np.array([
            market_state['volatility'],
            market_state['volume'],
            market_state['order_imbalance'],
            inventory_risk,
            market_state['toxic_flow_ratio']
        ]).reshape(1, -1)
        
        # Prédiction du spread optimal
        spread_multiplier = self.model.predict(features)[0][0]
        
        # Calcul du spread final
        optimal_spread = self.base_spread * (1 + spread_multiplier)
        
        return np.clip(optimal_spread, self.min_spread, self.max_spread)
    
    def update_model(self, 
                    spreads: List[float],
                    pnls: List[float]):
        """Met à jour le modèle avec les nouvelles données."""
        if len(spreads) < 2:
            return
        
        # Préparation des données
        X = np.array(spreads[:-1]).reshape(-1, 1)
        y = np.array(pnls[1:]).reshape(-1, 1)
        
        # Entraînement incrémental
        self.model.fit(
            X, y,
            epochs=1,
            verbose=0
        )

class ToxicFlowDetector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_history = deque(maxlen=window_size)
        self.flow_metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict:
        """Initialise les métriques de flux."""
        return {
            'toxic_trades': 0,
            'total_trades': 0,
            'toxic_volume': 0,
            'total_volume': 0,
            'adverse_selection': 0
        }
    
    def analyze_trade(self, trade: Dict) -> Dict:
        """Analyse un trade pour la toxicité."""
        # Ajout à l'historique
        self.trade_history.append(trade)
        
        # Calcul des métriques
        price_impact = self._calculate_price_impact(trade)
        order_flow_toxicity = self._calculate_flow_toxicity()
        adverse_selection = self._estimate_adverse_selection()
        
        # Mise à jour des métriques
        is_toxic = price_impact > trade['spread'] / 2
        if is_toxic:
            self.flow_metrics['toxic_trades'] += 1
            self.flow_metrics['toxic_volume'] += trade['volume']
        
        self.flow_metrics['total_trades'] += 1
        self.flow_metrics['total_volume'] += trade['volume']
        self.flow_metrics['adverse_selection'] = adverse_selection
        
        return {
            'is_toxic': is_toxic,
            'price_impact': price_impact,
            'flow_toxicity': order_flow_toxicity,
            'adverse_selection': adverse_selection
        }
    
    def _calculate_price_impact(self, trade: Dict) -> float:
        """Calcule l'impact prix d'un trade."""
        if len(self.trade_history) < 2:
            return 0
        
        pre_price = self.trade_history[-2]['price']
        post_price = trade['price']
        
        return abs(post_price - pre_price) / pre_price
    
    def _calculate_flow_toxicity(self) -> float:
        """Calcule la toxicité du flux d'ordres."""
        if not self.trade_history:
            return 0
        
        # VPIN (Volume-synchronized Probability of Informed Trading)
        buy_volume = sum(t['volume'] for t in self.trade_history if t['side'] == 'buy')
        total_volume = sum(t['volume'] for t in self.trade_history)
        
        if total_volume == 0:
            return 0
        
        volume_imbalance = abs(2 * buy_volume - total_volume) / total_volume
        return volume_imbalance
    
    def _estimate_adverse_selection(self) -> float:
        """Estime le coût de sélection adverse."""
        if len(self.trade_history) < 2:
            return 0
        
        # Calcul basé sur la corrélation entre le flux d'ordres et les mouvements de prix
        price_changes = [
            (t2['price'] - t1['price']) / t1['price']
            for t1, t2 in zip(self.trade_history[:-1], self.trade_history[1:])
        ]
        
        order_flows = [
            1 if t['side'] == 'buy' else -1
            for t in self.trade_history[:-1]
        ]
        
        if not price_changes or not order_flows:
            return 0
        
        correlation = np.corrcoef(price_changes, order_flows)[0, 1]
        return max(0, correlation)  # Only positive correlation indicates adverse selection

class AdaptiveMarketMaker:
    def __init__(self,
                 initial_capital: float,
                 risk_params: Dict):
        self.inventory_manager = InventoryManager(
            target_inventory=0,
            max_inventory=risk_params['max_inventory'],
            risk_aversion=risk_params['risk_aversion']
        )
        
        self.spread_optimizer = SpreadOptimizer(
            base_spread=risk_params['base_spread'],
            min_spread=risk_params['min_spread'],
            max_spread=risk_params['max_spread']
        )
        
        self.flow_detector = ToxicFlowDetector()
        
        self.state = MarketMakingState(
            inventory=0,
            cash=initial_capital,
            position_value=0,
            unrealized_pnl=0,
            realized_pnl=0,
            inventory_risk=0,
            spread_capture=0,
            order_flow_imbalance=0,
            toxic_flow_ratio=0,
            effective_spread=0
        )
    
    def generate_quotes(self,
                       market_state: Dict,
                       style: MarketMakingStyle = MarketMakingStyle.ADAPTIVE) -> Dict:
        """Génère des quotes optimisées."""
        # Analyse du marché
        inventory_risk = self.inventory_manager.calculate_inventory_risk(
            self.state.inventory,
            market_state['price'],
            market_state['volatility']
        )
        
        # Optimisation du spread
        base_spread = self.spread_optimizer.optimize_spread(
            market_state,
            inventory_risk
        )
        
        # Ajustements d'inventaire
        inventory_adjustments = self.inventory_manager.get_inventory_adjustments(
            self.state.inventory,
            market_state['price'],
            market_state['volatility']
        )
        
        # Ajustements selon le style
        spread_adjustments = self._apply_style_adjustments(
            base_spread,
            style,
            market_state
        )
        
        # Calcul des quotes finales
        mid_price = market_state['price']
        half_spread = base_spread / 2
        
        bid_price = mid_price - half_spread + inventory_adjustments['bid_adjustment']
        ask_price = mid_price + half_spread + inventory_adjustments['ask_adjustment']
        
        # Ajustements finaux selon le style
        bid_price += spread_adjustments['bid']
        ask_price += spread_adjustments['ask']
        
        return {
            'bid': bid_price,
            'ask': ask_price,
            'spread': ask_price - bid_price,
            'mid': mid_price,
            'inventory_risk': inventory_risk,
            'toxic_flow': self.state.toxic_flow_ratio
        }
    
    def _apply_style_adjustments(self,
                               base_spread: float,
                               style: MarketMakingStyle,
                               market_state: Dict) -> Dict[str, float]:
        """Applique des ajustements selon le style de market making."""
        adjustments = {'bid': 0, 'ask': 0}
        
        if style == MarketMakingStyle.PASSIVE:
            # Style passif: spreads plus larges, moins agressif
            adjustments['bid'] = -base_spread * 0.2
            adjustments['ask'] = base_spread * 0.2
            
        elif style == MarketMakingStyle.AGGRESSIVE:
            # Style agressif: spreads plus serrés
            adjustments['bid'] = base_spread * 0.1
            adjustments['ask'] = -base_spread * 0.1
            
        elif style == MarketMakingStyle.PREDATORY:
            # Style prédateur: exploite les opportunités de toxic flow
            if self.state.toxic_flow_ratio > 0.7:
                adjustments['bid'] = -base_spread * 0.3
                adjustments['ask'] = base_spread * 0.3
            
        elif style == MarketMakingStyle.ADAPTIVE:
            # Style adaptatif: s'adapte aux conditions de marché
            volatility_factor = market_state['volatility'] / 0.02  # normalized
            flow_factor = self.state.toxic_flow_ratio
            
            adjustments['bid'] = -base_spread * 0.1 * volatility_factor * (1 + flow_factor)
            adjustments['ask'] = base_spread * 0.1 * volatility_factor * (1 + flow_factor)
        
        return adjustments
    
    def update_state(self,
                    trades: List[Dict],
                    market_price: float):
        """Met à jour l'état du market maker."""
        for trade in trades:
            # Analyse de la toxicité
            flow_analysis = self.flow_detector.analyze_trade(trade)
            
            # Mise à jour des métriques
            self._update_metrics(trade, flow_analysis, market_price)
            
            # Mise à jour du spread optimizer
            self.spread_optimizer.update_model(
                [t['spread'] for t in trades],
                [t['pnl'] for t in trades]
            )
    
    def _update_metrics(self,
                       trade: Dict,
                       flow_analysis: Dict,
                       market_price: float):
        """Met à jour les métriques de performance."""
        # Mise à jour de l'inventaire
        if trade['side'] == 'buy':
            self.state.inventory += trade['volume']
            self.state.cash -= trade['volume'] * trade['price']
        else:
            self.state.inventory -= trade['volume']
            self.state.cash += trade['volume'] * trade['price']
        
        # Mise à jour des P&L
        self.state.position_value = self.state.inventory * market_price
        self.state.unrealized_pnl = (
            self.state.position_value + 
            self.state.cash - 
            self.state.realized_pnl
        )
        
        # Mise à jour des métriques de flow
        self.state.toxic_flow_ratio = flow_analysis['flow_toxicity']
        self.state.inventory_risk = self.inventory_manager.calculate_inventory_risk(
            self.state.inventory,
            market_price,
            trade.get('volatility', 0.02)
        )
        
        # Mise à jour du spread capture
        self.state.spread_capture = (
            trade['price'] - trade['mid_price']
            if trade['side'] == 'sell'
            else trade['mid_price'] - trade['price']
        ) / trade['mid_price']
