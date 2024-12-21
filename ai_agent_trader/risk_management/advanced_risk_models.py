import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import cvxopt
from arch import arch_model

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    volatility: float
    beta: float
    correlation: float
    tail_dependency: float
    regime_state: int
    regime_probability: float
    stress_var: float
    stress_es: float

class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42
        )
        self.regime_characteristics = {}
        
    def fit(self, returns: np.ndarray):
        """Entraîne le détecteur de régimes."""
        # Reshape pour GMM
        X = returns.reshape(-1, 1)
        
        # Fit du modèle
        self.gmm.fit(X)
        
        # Caractérisation des régimes
        labels = self.gmm.predict(X)
        probs = self.gmm.predict_proba(X)
        
        for i in range(self.n_regimes):
            regime_returns = returns[labels == i]
            self.regime_characteristics[i] = {
                'mean': np.mean(regime_returns),
                'std': np.std(regime_returns),
                'skew': stats.skew(regime_returns),
                'kurtosis': stats.kurtosis(regime_returns),
                'var_95': np.percentile(regime_returns, 5),
                'frequency': np.mean(labels == i)
            }
    
    def predict_regime(self, returns: np.ndarray) -> Tuple[int, float]:
        """Prédit le régime actuel et sa probabilité."""
        X = returns.reshape(-1, 1)
        regime = self.gmm.predict(X)[-1]
        probs = self.gmm.predict_proba(X)[-1]
        
        return regime, probs[regime]
    
    def get_regime_risk_factors(self, regime: int) -> Dict:
        """Retourne les facteurs de risque pour un régime."""
        if regime not in self.regime_characteristics:
            raise ValueError(f"Regime {regime} not found")
        
        return self.regime_characteristics[regime]

class ExtremeTailRiskAnalyzer:
    def __init__(self, threshold_percentile: float = 95):
        self.threshold_percentile = threshold_percentile
        self.gpd_params = None
        self.threshold = None
        
    def fit(self, returns: np.ndarray):
        """Ajuste le modèle de valeurs extrêmes."""
        # Calcul du seuil
        self.threshold = np.percentile(returns, self.threshold_percentile)
        
        # Sélection des excès
        exceedances = returns[returns > self.threshold] - self.threshold
        
        if len(exceedances) < 50:
            raise ValueError("Not enough extreme values for fitting")
        
        # Fit de la GPD
        self.gpd_params = self._fit_gpd(exceedances)
    
    def _fit_gpd(self, exceedances: np.ndarray) -> Dict:
        """Ajuste une distribution de Pareto généralisée."""
        def neg_log_likelihood(params):
            xi, beta = params
            n = len(exceedances)
            if beta <= 0:
                return np.inf
            if xi == 0:
                return n * np.log(beta) + np.sum(exceedances) / beta
            else:
                if np.any(1 + xi * exceedances / beta <= 0):
                    return np.inf
                return (n * np.log(beta) + 
                       (1 + 1/xi) * np.sum(np.log(1 + xi * exceedances / beta)))
        
        # Optimisation
        result = minimize(
            neg_log_likelihood,
            x0=[0.1, np.std(exceedances)],
            method='Nelder-Mead'
        )
        
        return {
            'xi': result.x[0],
            'beta': result.x[1]
        }
    
    def estimate_tail_risk(self, confidence: float = 0.99) -> Dict:
        """Estime les mesures de risque de queue."""
        if self.gpd_params is None:
            raise ValueError("Model not fitted")
        
        xi = self.gpd_params['xi']
        beta = self.gpd_params['beta']
        
        # VaR
        p = (1 - confidence) * (1 - self.threshold_percentile/100)
        if xi == 0:
            var = self.threshold - beta * np.log(p)
        else:
            var = self.threshold + (beta/xi) * (p**(-xi) - 1)
        
        # ES
        if xi >= 1:
            es = np.inf
        else:
            es = var/(1-xi) + (beta - xi*self.threshold)/(1-xi)
        
        return {
            'var': var,
            'es': es,
            'tail_index': xi
        }

class DynamicVolatilityModel:
    def __init__(self):
        self.model = None
        self.last_vol = None
        
    def fit(self, returns: np.ndarray):
        """Ajuste un modèle GARCH."""
        self.model = arch_model(
            returns,
            vol='Garch',
            p=1, q=1,
            dist='skewt'
        )
        self.result = self.model.fit(disp='off')
        
    def forecast_volatility(self, horizon: int = 1) -> np.ndarray:
        """Prévoit la volatilité future."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        forecast = self.result.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1])
    
    def get_current_vol(self) -> float:
        """Retourne la volatilité actuelle."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return np.sqrt(self.result.conditional_volatility[-1])

class CopulaBasedDependenceAnalyzer:
    def __init__(self):
        self.copula = None
        self.marginals = {}
        
    def fit(self, data: pd.DataFrame):
        """Ajuste une copule aux données."""
        # Transformation en rangs uniformes
        U = pd.DataFrame()
        for col in data.columns:
            self.marginals[col] = stats.norm.fit(data[col])
            U[col] = stats.norm.cdf(data[col], *self.marginals[col])
        
        # Ajustement de la copule gaussienne
        self.copula = stats.multivariate_normal.fit(
            stats.norm.ppf(U)
        )
    
    def estimate_tail_dependence(self) -> Dict:
        """Estime la dépendance de queue."""
        if self.copula is None:
            raise ValueError("Copula not fitted")
        
        # Simulation pour l'estimation
        n_sim = 10000
        sim_data = stats.multivariate_normal.rvs(
            mean=self.copula[0],
            cov=self.copula[1],
            size=n_sim
        )
        
        # Transformation en uniformes
        U = stats.norm.cdf(sim_data)
        
        # Estimation de la dépendance de queue
        q = 0.05
        lower_tail = np.mean(
            (U[:, 0] < q) & (U[:, 1] < q)
        ) / q
        
        upper_tail = np.mean(
            (U[:, 0] > 1-q) & (U[:, 1] > 1-q)
        ) / q
        
        return {
            'lower_tail': lower_tail,
            'upper_tail': upper_tail
        }

class AdvancedRiskManager:
    def __init__(self,
                 confidence_levels: List[float] = [0.95, 0.99],
                 n_regimes: int = 3):
        self.confidence_levels = confidence_levels
        
        # Composants
        self.regime_detector = MarketRegimeDetector(n_regimes)
        self.tail_analyzer = ExtremeTailRiskAnalyzer()
        self.vol_model = DynamicVolatilityModel()
        self.dependence_analyzer = CopulaBasedDependenceAnalyzer()
        
        # État
        self.current_regime = None
        self.regime_prob = None
        
    def update_risk_models(self, 
                          returns: np.ndarray,
                          market_data: Optional[pd.DataFrame] = None):
        """Met à jour tous les modèles de risque."""
        # Mise à jour du détecteur de régimes
        self.regime_detector.fit(returns)
        self.current_regime, self.regime_prob = self.regime_detector.predict_regime(returns)
        
        # Mise à jour de l'analyseur de queues
        self.tail_analyzer.fit(returns)
        
        # Mise à jour du modèle de volatilité
        self.vol_model.fit(returns)
        
        # Mise à jour de l'analyseur de dépendance
        if market_data is not None:
            self.dependence_analyzer.fit(market_data)
    
    def compute_risk_metrics(self, 
                           returns: np.ndarray,
                           market_returns: Optional[np.ndarray] = None) -> RiskMetrics:
        """Calcule toutes les métriques de risque."""
        # VaR et ES standards
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        es_95 = np.mean(returns[returns < var_95])
        es_99 = np.mean(returns[returns < var_99])
        
        # Volatilité conditionnelle
        volatility = self.vol_model.get_current_vol()
        
        # Beta et corrélation
        if market_returns is not None:
            beta = np.cov(returns, market_returns)[0,1] / np.var(market_returns)
            correlation = np.corrcoef(returns, market_returns)[0,1]
        else:
            beta = correlation = np.nan
        
        # Dépendance de queue
        tail_metrics = self.tail_analyzer.estimate_tail_risk()
        tail_dependency = tail_metrics['tail_index']
        
        # Métriques de stress
        regime_factors = self.regime_detector.get_regime_risk_factors(self.current_regime)
        stress_var = regime_factors['var_95']
        stress_es = np.mean(returns[returns < stress_var])
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            volatility=volatility,
            beta=beta,
            correlation=correlation,
            tail_dependency=tail_dependency,
            regime_state=self.current_regime,
            regime_probability=self.regime_prob,
            stress_var=stress_var,
            stress_es=stress_es
        )
    
    def optimize_portfolio(self,
                         returns: pd.DataFrame,
                         constraints: Optional[Dict] = None) -> Dict:
        """Optimise le portfolio avec contraintes de risque."""
        n_assets = returns.shape[1]
        
        # Matrices de covariance conditionnelle
        returns_array = returns.values
        cov_matrix = np.cov(returns_array.T)
        
        # Rendements espérés par régime
        expected_returns = []
        for i in range(self.regime_detector.n_regimes):
            regime_factors = self.regime_detector.get_regime_risk_factors(i)
            expected_returns.append(regime_factors['mean'])
        
        expected_returns = np.array(expected_returns)
        
        # Fonction objectif: maximiser le ratio de Sharpe conditionnel au régime
        def objective(weights):
            portfolio_returns = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_returns/portfolio_vol
        
        # Contraintes
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # somme = 1
        ]
        
        if constraints:
            if 'min_weight' in constraints:
                constraints_list.append(
                    {'type': 'ineq', 'fun': lambda x: x - constraints['min_weight']}
                )
            if 'max_weight' in constraints:
                constraints_list.append(
                    {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}
                )
        
        # Optimisation
        result = minimize(
            objective,
            x0=np.ones(n_assets)/n_assets,
            constraints=constraints_list,
            bounds=[(0,1) for _ in range(n_assets)]
        )
        
        return {
            'weights': result.x,
            'sharpe': -result.fun,
            'success': result.success
        }
    
    def compute_stress_scenarios(self,
                               portfolio_weights: np.ndarray,
                               returns: pd.DataFrame,
                               n_scenarios: int = 1000) -> Dict:
        """Génère et analyse des scénarios de stress."""
        # Paramètres du régime actuel
        regime_factors = self.regime_detector.get_regime_risk_factors(self.current_regime)
        
        # Simulation de scénarios
        scenarios = np.random.normal(
            loc=regime_factors['mean'],
            scale=regime_factors['std'],
            size=(n_scenarios, returns.shape[1])
        )
        
        # Application des poids du portfolio
        portfolio_scenarios = np.dot(scenarios, portfolio_weights)
        
        # Analyse des scénarios
        worst_loss = np.min(portfolio_scenarios)
        var_stress = np.percentile(portfolio_scenarios, 1)
        es_stress = np.mean(portfolio_scenarios[portfolio_scenarios < var_stress])
        
        return {
            'worst_case': worst_loss,
            'var_stress': var_stress,
            'es_stress': es_stress,
            'scenarios': portfolio_scenarios
        }
