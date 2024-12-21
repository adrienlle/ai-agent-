import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    DEGEN = "DEGEN"

@dataclass
class PositionConfig:
    max_portfolio_risk: float
    position_size: float
    entry_points: List[float]
    exit_points: List[float]
    stop_loss: float
    leverage: float

class MoneyManager:
    """Gestionnaire sophistiqué de money management et risk management."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions = []
        self.position_history = []
        self.risk_metrics = self._init_risk_metrics()
    
    def _init_risk_metrics(self) -> Dict:
        """Initialise les métriques de risque du portfolio."""
        return {
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'risk_adjusted_return': 0.0,
            'kelly_criterion': 0.0
        }
    
    def calculate_position_size(self, opportunity: Dict, risk_level: RiskLevel) -> PositionConfig:
        """Calcule la taille optimale de position basée sur plusieurs facteurs."""
        
        # Paramètres de base selon le niveau de risque
        risk_params = {
            RiskLevel.LOW: {'max_risk': 0.01, 'max_leverage': 1.0},
            RiskLevel.MEDIUM: {'max_risk': 0.02, 'max_leverage': 2.0},
            RiskLevel.HIGH: {'max_risk': 0.05, 'max_leverage': 3.0},
            RiskLevel.DEGEN: {'max_risk': 0.10, 'max_leverage': 5.0}
        }[risk_level]
        
        # Calcul du Kelly Criterion modifié
        win_prob = opportunity['analysis']['buy_confidence'] / 100
        win_loss_ratio = abs(opportunity['analysis']['predicted_roi_24h'] / 15)  # Assume 15% stop loss
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly = max(0, min(kelly * 0.5, risk_params['max_risk']))  # Kelly fractionnel
        
        # Ajustement par la volatilité
        volatility_factor = random.uniform(0.7, 1.0)  # Simulé pour la démo
        position_size = self.current_capital * kelly * volatility_factor
        
        # Génération des points d'entrée échelonnés
        entry_points = self._generate_entry_points(opportunity['token']['price'], 3)
        
        # Génération des take profits optimisés
        exit_points = self._generate_exit_points(
            opportunity['token']['price'],
            opportunity['analysis']['predicted_roi_24h']
        )
        
        # Calcul du stop loss dynamique
        stop_loss = self._calculate_dynamic_stop_loss(
            opportunity['token']['price'],
            opportunity['analysis']['volatility_score'] if 'volatility_score' in opportunity['analysis'] else 50
        )
        
        # Calcul du levier optimal
        leverage = self._calculate_optimal_leverage(
            risk_params['max_leverage'],
            opportunity['analysis']['risk_score'] / 100
        )
        
        return PositionConfig(
            max_portfolio_risk=risk_params['max_risk'],
            position_size=position_size,
            entry_points=entry_points,
            exit_points=exit_points,
            stop_loss=stop_loss,
            leverage=leverage
        )
    
    def _generate_entry_points(self, base_price: float, num_points: int) -> List[float]:
        """Génère des points d'entrée échelonnés."""
        spread = np.linspace(-0.02, 0.02, num_points)
        return [base_price * (1 + s) for s in spread]
    
    def _generate_exit_points(self, entry_price: float, predicted_roi: float) -> List[float]:
        """Génère des points de sortie optimisés."""
        target_price = entry_price * (1 + predicted_roi/100)
        return [
            entry_price * 1.1,  # TP1
            (entry_price + target_price) / 2,  # TP2
            target_price,  # TP3
            target_price * 1.2  # TP4 (bonus)
        ]
    
    def _calculate_dynamic_stop_loss(self, entry_price: float, volatility_score: float) -> float:
        """Calcule un stop loss dynamique basé sur la volatilité."""
        base_stop = 0.15  # 15% stop loss de base
        volatility_adjustment = (volatility_score / 100) * 0.1  # +/- 10% selon la volatilité
        return entry_price * (1 - (base_stop + volatility_adjustment))
    
    def _calculate_optimal_leverage(self, max_leverage: float, risk_score: float) -> float:
        """Calcule le levier optimal basé sur le score de risque."""
        return max_leverage * (1 - risk_score)  # Réduit le levier quand le risque augmente
    
    def update_portfolio_metrics(self, trade_result: Dict):
        """Met à jour les métriques du portfolio après un trade."""
        self.position_history.append(trade_result)
        
        # Mise à jour du capital
        pnl = trade_result.get('pnl', 0)
        self.current_capital += pnl
        
        # Calcul du drawdown
        peak_capital = max(self.initial_capital, self.current_capital)
        current_drawdown = (peak_capital - self.current_capital) / peak_capital
        self.risk_metrics['max_drawdown'] = max(self.risk_metrics['max_drawdown'], current_drawdown)
        
        # Mise à jour des autres métriques
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Met à jour les métriques de performance du portfolio."""
        if not self.position_history:
            return
        
        # Calcul du win rate
        wins = sum(1 for trade in self.position_history if trade.get('pnl', 0) > 0)
        self.risk_metrics['win_rate'] = wins / len(self.position_history)
        
        # Calcul de la volatilité (simulé pour la démo)
        returns = [trade.get('roi', 0) for trade in self.position_history]
        self.risk_metrics['volatility'] = np.std(returns) if returns else 0
        
        # Calcul du ratio de Sharpe (simulé)
        risk_free_rate = 0.02  # 2% taux sans risque
        if self.risk_metrics['volatility'] > 0:
            avg_return = np.mean(returns) if returns else 0
            self.risk_metrics['sharpe_ratio'] = (avg_return - risk_free_rate) / self.risk_metrics['volatility']
        
        # Mise à jour du Kelly Criterion global
        if self.risk_metrics['win_rate'] > 0:
            avg_win = np.mean([t['roi'] for t in self.position_history if t.get('pnl', 0) > 0]) if wins else 0
            avg_loss = abs(np.mean([t['roi'] for t in self.position_history if t.get('pnl', 0) <= 0])) if len(self.position_history) > wins else 1
            self.risk_metrics['kelly_criterion'] = (self.risk_metrics['win_rate'] * avg_win - (1 - self.risk_metrics['win_rate'])) / avg_loss
    
    def get_portfolio_summary(self) -> Dict:
        """Retourne un résumé détaillé du portfolio."""
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'open_positions': len(self.open_positions),
            'total_trades': len(self.position_history),
            'risk_metrics': self.risk_metrics
        }

class PortfolioOptimizer:
    """Optimiseur de portfolio utilisant des techniques avancées."""
    
    def __init__(self, positions: List[Dict]):
        self.positions = positions
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Optimise l'allocation du portfolio selon plusieurs stratégies."""
        # Simulation d'une optimisation sophistiquée
        allocations = {}
        total_weight = 0
        
        for pos in self.positions:
            # Calcul du score d'allocation basé sur plusieurs facteurs
            score = (
                pos['analysis']['buy_confidence'] * 0.3 +
                pos['analysis']['risk_score'] * 0.2 +
                random.uniform(0, 50) * 0.5  # Facteur aléatoire pour la démo
            )
            allocations[pos['token']['symbol']] = score
            total_weight += score
        
        # Normalisation des allocations
        if total_weight > 0:
            allocations = {k: v/total_weight for k, v in allocations.items()}
        
        return allocations
