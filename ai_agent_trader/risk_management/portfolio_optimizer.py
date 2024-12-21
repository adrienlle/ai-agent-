import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
import cvxopt
from cvxopt import matrix, solvers
import empyrical

class RiskMetric(Enum):
    VOLATILITY = "VOLATILITY"
    VALUE_AT_RISK = "VALUE_AT_RISK"
    EXPECTED_SHORTFALL = "EXPECTED_SHORTFALL"
    MAXIMUM_DRAWDOWN = "MAXIMUM_DRAWDOWN"
    BETA = "BETA"
    CORRELATION = "CORRELATION"
    TAIL_DEPENDENCE = "TAIL_DEPENDENCE"

@dataclass
class PortfolioConstraints:
    min_weights: np.ndarray
    max_weights: np.ndarray
    target_return: float
    max_volatility: float
    max_drawdown: float
    beta_range: Tuple[float, float]
    sector_constraints: Dict[str, Tuple[float, float]]
    turnover_limit: float

class AdvancedPortfolioOptimizer:
    def __init__(self,
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Optimiseur de portfolio sophistiqué utilisant multiple métriques de risque.
        
        Args:
            returns: DataFrame des rendements historiques
            risk_free_rate: Taux sans risque annualisé
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
        # Calcul des métriques de base
        self.mean_returns = returns.mean()
        self.cov_matrix = self._compute_robust_covariance()
        
        # Métriques de risque avancées
        self.var_95 = self._compute_historical_var()
        self.es_95 = self._compute_expected_shortfall()
        self.max_drawdowns = self._compute_max_drawdowns()
        self.tail_dependencies = self._compute_tail_dependencies()
    
    def _compute_robust_covariance(self) -> np.ndarray:
        """Calcule une matrice de covariance robuste."""
        # Méthode de Ledoit-Wolf pour l'estimation robuste
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        return lw.fit(self.returns).covariance_
    
    def _compute_historical_var(self, confidence: float = 0.95) -> np.ndarray:
        """Calcule la VaR historique pour chaque actif."""
        return np.percentile(self.returns, (1 - confidence) * 100, axis=0)
    
    def _compute_expected_shortfall(self, confidence: float = 0.95) -> np.ndarray:
        """Calcule l'Expected Shortfall pour chaque actif."""
        var = self._compute_historical_var(confidence)
        return np.array([
            self.returns[self.returns[col] <= var[i]][col].mean()
            for i, col in enumerate(self.returns.columns)
        ])
    
    def _compute_max_drawdowns(self) -> np.ndarray:
        """Calcule le drawdown maximum pour chaque actif."""
        return np.array([
            empyrical.max_drawdown(self.returns[col])
            for col in self.returns.columns
        ])
    
    def _compute_tail_dependencies(self) -> np.ndarray:
        """Calcule les dépendances de queue entre les actifs."""
        n = self.n_assets
        tail_dep = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Calcul de la dépendance de queue inférieure
                x = self.returns.iloc[:, i]
                y = self.returns.iloc[:, j]
                q = 0.05  # 5% quantile
                
                chi = np.mean(
                    (x <= np.quantile(x, q)) & 
                    (y <= np.quantile(y, q))
                ) / q
                
                tail_dep[i, j] = tail_dep[j, i] = chi
        
        return tail_dep
    
    def optimize(self,
                constraints: PortfolioConstraints,
                objective: str = "SHARPE",
                risk_aversion: float = 2.0) -> Dict:
        """
        Optimise le portfolio selon différents objectifs et contraintes.
        
        Args:
            constraints: Contraintes du portfolio
            objective: Objectif d'optimisation ("SHARPE", "MIN_RISK", "MAX_RETURN", "RISK_PARITY")
            risk_aversion: Coefficient d'aversion au risque pour l'utilité moyenne-variance
            
        Returns:
            Dict contenant les poids optimaux et les métriques du portfolio
        """
        if objective == "SHARPE":
            weights = self._optimize_sharpe(constraints)
        elif objective == "MIN_RISK":
            weights = self._optimize_min_risk(constraints)
        elif objective == "MAX_RETURN":
            weights = self._optimize_max_return(constraints)
        elif objective == "RISK_PARITY":
            weights = self._optimize_risk_parity(constraints)
        else:
            raise ValueError(f"Objectif d'optimisation inconnu: {objective}")
        
        # Calcul des métriques du portfolio
        metrics = self._compute_portfolio_metrics(weights)
        
        return {
            'weights': weights,
            'metrics': metrics,
            'decomposition': self._risk_decomposition(weights)
        }
    
    def _optimize_sharpe(self, constraints: PortfolioConstraints) -> np.ndarray:
        """Optimisation du ratio de Sharpe."""
        def objective(weights):
            portfolio_return = np.sum(self.mean_returns * weights) * 252
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimisation
        
        constraints_list = self._create_optimization_constraints(constraints)
        
        result = minimize(
            objective,
            x0=np.array([1/self.n_assets] * self.n_assets),
            method='SLSQP',
            constraints=constraints_list,
            bounds=[(min_w, max_w) for min_w, max_w in zip(
                constraints.min_weights,
                constraints.max_weights
            )]
        )
        
        return result.x
    
    def _optimize_min_risk(self, constraints: PortfolioConstraints) -> np.ndarray:
        """Optimisation de la volatilité minimale."""
        def objective(weights):
            return np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
        
        constraints_list = self._create_optimization_constraints(constraints)
        
        result = minimize(
            objective,
            x0=np.array([1/self.n_assets] * self.n_assets),
            method='SLSQP',
            constraints=constraints_list,
            bounds=[(min_w, max_w) for min_w, max_w in zip(
                constraints.min_weights,
                constraints.max_weights
            )]
        )
        
        return result.x
    
    def _optimize_risk_parity(self, constraints: PortfolioConstraints) -> np.ndarray:
        """Optimisation Risk Parity (parité des risques)."""
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )
            marginal_risk = np.dot(self.cov_matrix * 252, weights) / portfolio_vol
            risk_contrib = weights * marginal_risk
            return risk_contrib
        
        def objective(weights):
            risk_contrib = risk_contribution(weights)
            target_risk = 1.0 / self.n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        constraints_list = self._create_optimization_constraints(constraints)
        
        result = minimize(
            objective,
            x0=np.array([1/self.n_assets] * self.n_assets),
            method='SLSQP',
            constraints=constraints_list,
            bounds=[(min_w, max_w) for min_w, max_w in zip(
                constraints.min_weights,
                constraints.max_weights
            )]
        )
        
        return result.x
    
    def _create_optimization_constraints(self,
                                      constraints: PortfolioConstraints) -> List:
        """Crée la liste des contraintes pour l'optimisation."""
        constraint_list = []
        
        # Contrainte de somme des poids
        constraint_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        # Contrainte de rendement minimum
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(self.mean_returns * x) * 252 - constraints.target_return
            })
        
        # Contrainte de volatilité maximum
        if constraints.max_volatility is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.max_volatility - np.sqrt(
                    np.dot(x.T, np.dot(self.cov_matrix * 252, x))
                )
            })
        
        # Contraintes sectorielles
        for sector, (min_weight, max_weight) in constraints.sector_constraints.items():
            sector_assets = self._get_sector_assets(sector)
            if min_weight is not None:
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: np.sum(x[sector_assets]) - min_weight
                })
            if max_weight is not None:
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x: max_weight - np.sum(x[sector_assets])
                })
        
        return constraint_list
    
    def _compute_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calcule les métriques complètes du portfolio."""
        portfolio_returns = np.dot(self.returns, weights)
        
        metrics = {
            'return': np.sum(self.mean_returns * weights) * 252,
            'volatility': np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            ),
            'sharpe': empyrical.sharpe_ratio(
                portfolio_returns,
                risk_free=self.risk_free_rate/252
            ),
            'sortino': empyrical.sortino_ratio(
                portfolio_returns,
                risk_free=self.risk_free_rate/252
            ),
            'max_drawdown': empyrical.max_drawdown(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'es_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'skewness': pd.Series(portfolio_returns).skew(),
            'kurtosis': pd.Series(portfolio_returns).kurtosis(),
            'calmar_ratio': empyrical.calmar_ratio(portfolio_returns),
            'omega_ratio': empyrical.omega_ratio(portfolio_returns)
        }
        
        return metrics
    
    def _risk_decomposition(self, weights: np.ndarray) -> Dict:
        """Décompose le risque du portfolio."""
        # Volatilité totale
        portfolio_vol = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
        )
        
        # Contribution marginale au risque
        marginal_risk = np.dot(self.cov_matrix * 252, weights) / portfolio_vol
        
        # Contribution au risque
        risk_contribution = weights * marginal_risk
        
        # Contribution en pourcentage
        pct_risk_contribution = risk_contribution / portfolio_vol
        
        # Décomposition par composante
        decomposition = {
            'marginal_risk': marginal_risk,
            'risk_contribution': risk_contribution,
            'pct_risk_contribution': pct_risk_contribution,
            'diversification_ratio': portfolio_vol / np.sum(weights * np.sqrt(np.diag(self.cov_matrix)) * np.sqrt(252))
        }
        
        return decomposition
    
    def stress_test(self,
                   weights: np.ndarray,
                   scenarios: Dict[str, Dict[str, float]]) -> Dict:
        """
        Effectue des tests de stress sur le portfolio.
        
        Args:
            weights: Poids du portfolio
            scenarios: Dictionnaire des scénarios de stress
            
        Returns:
            Dict des résultats des tests de stress
        """
        results = {}
        
        for scenario_name, changes in scenarios.items():
            # Ajustement des rendements moyens
            adjusted_returns = self.mean_returns.copy()
            for asset, change in changes.items():
                if asset in self.returns.columns:
                    adjusted_returns[asset] += change
            
            # Calcul du rendement du portfolio sous stress
            stress_return = np.sum(adjusted_returns * weights) * 252
            
            # Calcul de la VaR stressée
            stress_var = self._compute_stress_var(weights, changes)
            
            results[scenario_name] = {
                'portfolio_return': stress_return,
                'var_95': stress_var,
                'impact': stress_return - np.sum(self.mean_returns * weights) * 252
            }
        
        return results
    
    def _compute_stress_var(self,
                          weights: np.ndarray,
                          stress_changes: Dict[str, float],
                          confidence: float = 0.95) -> float:
        """Calcule la VaR sous conditions de stress."""
        # Simulation historique avec ajustements de stress
        stress_returns = self.returns.copy()
        
        for asset, change in stress_changes.items():
            if asset in stress_returns.columns:
                stress_returns[asset] += change
        
        portfolio_returns = np.dot(stress_returns, weights)
        return np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def rebalance(self,
                 current_weights: np.ndarray,
                 target_weights: np.ndarray,
                 constraints: PortfolioConstraints) -> Dict:
        """
        Optimise le rebalancement du portfolio.
        
        Args:
            current_weights: Poids actuels
            target_weights: Poids cibles
            constraints: Contraintes de rebalancement
            
        Returns:
            Dict contenant les trades optimaux
        """
        def objective(x):
            # Minimise les coûts de transaction
            return np.sum(np.abs(x - current_weights))
        
        constraints_list = self._create_optimization_constraints(constraints)
        
        # Contrainte de turnover
        if constraints.turnover_limit:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: constraints.turnover_limit - np.sum(np.abs(x - current_weights))
            })
        
        result = minimize(
            objective,
            x0=target_weights,
            method='SLSQP',
            constraints=constraints_list,
            bounds=[(min_w, max_w) for min_w, max_w in zip(
                constraints.min_weights,
                constraints.max_weights
            )]
        )
        
        trades = result.x - current_weights
        
        return {
            'optimal_weights': result.x,
            'trades': trades,
            'turnover': np.sum(np.abs(trades)),
            'transaction_cost': self._estimate_transaction_costs(trades)
        }
    
    def _estimate_transaction_costs(self, trades: np.ndarray) -> float:
        """Estime les coûts de transaction."""
        # Simulation simple avec spread + impact
        SPREAD = 0.0020  # 20 bps
        IMPACT = 0.0010  # 10 bps
        
        return np.sum(np.abs(trades)) * (SPREAD + IMPACT)
