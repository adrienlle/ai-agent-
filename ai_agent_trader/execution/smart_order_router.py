import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import random
from datetime import datetime, timedelta
import pandas as pd
from collections import deque

class ExecutionAlgorithm(Enum):
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"
    ADAPTIVE = "ADAPTIVE"
    ICEBERG = "ICEBERG"
    SNIPER = "SNIPER"
    DARK_ICEBERG = "DARK_ICEBERG"
    LIQUIDITY_SEEKING = "LIQUIDITY_SEEKING"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    CONDITIONAL = "CONDITIONAL"
    FILL_OR_KILL = "FILL_OR_KILL"
    IMMEDIATE_OR_CANCEL = "IMMEDIATE_OR_CANCEL"
    POST_ONLY = "POST_ONLY"

@dataclass
class ExecutionMetrics:
    slippage: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float
    execution_quality: float
    fill_rate: float
    average_spread: float
    realized_volatility: float

@dataclass
class MarketState:
    bid: float
    ask: float
    last_price: float
    bid_size: float
    ask_size: float
    volume_24h: float
    volatility_24h: float
    order_book_imbalance: float
    market_impact_estimate: float
    liquidity_score: float

class SmartOrderRouter:
    def __init__(self,
                 exchanges: List[Dict],
                 execution_params: Dict):
        """
        Routeur d'ordres intelligent avec exécution sophistiquée.
        
        Args:
            exchanges: Liste des exchanges disponibles avec leurs paramètres
            execution_params: Paramètres d'exécution
        """
        self.exchanges = exchanges
        self.params = execution_params
        self.execution_history = []
        self.market_impact_model = self._initialize_impact_model()
        self.order_book_cache = {}
        self.execution_metrics = {}
        
        # File d'attente pour le rate limiting
        self.rate_limit_queues = {
            exchange['name']: deque(maxlen=exchange['rate_limit'])
            for exchange in exchanges
        }
    
    def _initialize_impact_model(self):
        """Initialise le modèle d'impact de marché."""
        class MarketImpactModel:
            def __init__(self):
                self.alpha = 0.1
                self.beta = 0.4
                self.gamma = 0.2
                
            def estimate_impact(self,
                              order_size: float,
                              market_state: MarketState) -> float:
                """Estime l'impact de marché d'un ordre."""
                normalized_size = order_size / market_state.volume_24h
                spread = (market_state.ask - market_state.bid) / market_state.last_price
                
                temporary_impact = (
                    self.alpha * normalized_size ** self.beta *
                    (1 + self.gamma * market_state.volatility_24h) *
                    spread
                )
                
                permanent_impact = temporary_impact * 0.3
                
                return {
                    'temporary': temporary_impact,
                    'permanent': permanent_impact,
                    'total': temporary_impact + permanent_impact
                }
        
        return MarketImpactModel()
    
    async def execute_order(self,
                          symbol: str,
                          side: str,
                          size: float,
                          algorithm: ExecutionAlgorithm,
                          constraints: Dict) -> Dict:
        """
        Exécute un ordre de manière optimale.
        
        Args:
            symbol: Symbole à trader
            side: 'BUY' ou 'SELL'
            size: Taille totale à exécuter
            algorithm: Algorithme d'exécution
            constraints: Contraintes d'exécution
            
        Returns:
            Dict contenant les résultats d'exécution
        """
        # Initialisation
        start_time = time.time()
        execution_state = {
            'filled_size': 0,
            'average_price': 0,
            'slippage': 0,
            'remaining_size': size
        }
        
        # Sélection de la stratégie d'exécution
        if algorithm == ExecutionAlgorithm.ADAPTIVE:
            execution_func = self._adaptive_execution
        elif algorithm == ExecutionAlgorithm.LIQUIDITY_SEEKING:
            execution_func = self._liquidity_seeking_execution
        else:
            execution_func = self._standard_execution
        
        try:
            # Boucle principale d'exécution
            while execution_state['remaining_size'] > 0:
                # Mise à jour de l'état du marché
                market_state = await self._get_market_state(symbol)
                
                # Vérification des conditions d'arrêt
                if self._should_stop_execution(market_state, constraints):
                    break
                
                # Exécution d'une tranche
                execution_result = await execution_func(
                    symbol=symbol,
                    side=side,
                    remaining_size=execution_state['remaining_size'],
                    market_state=market_state,
                    constraints=constraints
                )
                
                # Mise à jour de l'état d'exécution
                self._update_execution_state(execution_state, execution_result)
                
                # Attente adaptative
                await self._adaptive_wait(market_state)
        
        except Exception as e:
            print(f"Erreur d'exécution: {e}")
            # Tentative de récupération ou annulation si nécessaire
            await self._handle_execution_error(execution_state)
        
        finally:
            # Calcul des métriques finales
            execution_metrics = self._calculate_execution_metrics(
                execution_state,
                start_time,
                market_state
            )
            
            return {
                'execution_state': execution_state,
                'metrics': execution_metrics,
                'market_state': market_state
            }
    
    async def _adaptive_execution(self,
                                symbol: str,
                                side: str,
                                remaining_size: float,
                                market_state: MarketState,
                                constraints: Dict) -> Dict:
        """Exécution adaptative basée sur les conditions de marché."""
        # Calcul de la taille optimale de la tranche
        participation_rate = self._calculate_optimal_participation(
            market_state,
            remaining_size
        )
        
        tranche_size = min(
            remaining_size,
            market_state.volume_24h * participation_rate
        )
        
        # Sélection dynamique de la stratégie
        if market_state.volatility_24h > constraints.get('high_volatility_threshold', 0.02):
            return await self._aggressive_execution(
                symbol, side, tranche_size, market_state
            )
        elif market_state.liquidity_score < constraints.get('low_liquidity_threshold', 0.3):
            return await self._passive_execution(
                symbol, side, tranche_size, market_state
            )
        else:
            return await self._balanced_execution(
                symbol, side, tranche_size, market_state
            )
    
    async def _liquidity_seeking_execution(self,
                                         symbol: str,
                                         side: str,
                                         remaining_size: float,
                                         market_state: MarketState,
                                         constraints: Dict) -> Dict:
        """Exécution cherchant la meilleure liquidité disponible."""
        # Analyse de la liquidité sur tous les exchanges
        liquidity_map = await self._analyze_cross_exchange_liquidity(symbol)
        
        # Optimisation de la distribution des ordres
        order_distribution = self._optimize_order_distribution(
            liquidity_map,
            remaining_size,
            constraints
        )
        
        # Exécution parallèle sur multiple exchanges
        execution_tasks = []
        for exchange, size in order_distribution.items():
            if size > 0:
                task = self._execute_on_exchange(
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    size=size,
                    market_state=market_state
                )
                execution_tasks.append(task)
        
        # Attente des résultats
        results = await asyncio.gather(*execution_tasks)
        
        # Agrégation des résultats
        return self._aggregate_execution_results(results)
    
    async def _analyze_cross_exchange_liquidity(self,
                                              symbol: str) -> Dict[str, Dict]:
        """Analyse la liquidité disponible sur tous les exchanges."""
        liquidity_map = {}
        
        for exchange in self.exchanges:
            try:
                # Récupération du carnet d'ordres
                order_book = await self._get_order_book(exchange['name'], symbol)
                
                # Calcul des métriques de liquidité
                liquidity_metrics = self._calculate_liquidity_metrics(order_book)
                
                # Évaluation des coûts d'exécution
                execution_costs = self._estimate_execution_costs(
                    exchange['name'],
                    order_book
                )
                
                liquidity_map[exchange['name']] = {
                    'metrics': liquidity_metrics,
                    'costs': execution_costs,
                    'score': self._calculate_liquidity_score(
                        liquidity_metrics,
                        execution_costs
                    )
                }
            
            except Exception as e:
                print(f"Erreur lors de l'analyse de {exchange['name']}: {e}")
                continue
        
        return liquidity_map
    
    def _optimize_order_distribution(self,
                                   liquidity_map: Dict,
                                   total_size: float,
                                   constraints: Dict) -> Dict[str, float]:
        """Optimise la distribution des ordres entre les exchanges."""
        distribution = {}
        
        # Normalisation des scores de liquidité
        total_score = sum(data['score'] for data in liquidity_map.values())
        
        if total_score > 0:
            # Distribution proportionnelle à la liquidité
            for exchange, data in liquidity_map.items():
                base_allocation = (data['score'] / total_score) * total_size
                
                # Ajustement pour les contraintes
                max_size = min(
                    base_allocation,
                    data['metrics']['available_liquidity'],
                    constraints.get('max_exchange_size', float('inf'))
                )
                
                distribution[exchange] = max_size
            
            # Normalisation finale
            total_allocated = sum(distribution.values())
            if total_allocated > total_size:
                scale_factor = total_size / total_allocated
                distribution = {
                    k: v * scale_factor for k, v in distribution.items()
                }
        
        return distribution
    
    async def _execute_on_exchange(self,
                                 exchange: str,
                                 symbol: str,
                                 side: str,
                                 size: float,
                                 market_state: MarketState) -> Dict:
        """Exécute un ordre sur un exchange spécifique."""
        # Vérification du rate limiting
        await self._check_rate_limit(exchange)
        
        try:
            # Sélection de la stratégie d'ordre optimale
            order_strategy = self._select_order_strategy(
                exchange,
                market_state,
                size
            )
            
            # Placement de l'ordre
            order_result = await self._place_order(
                exchange=exchange,
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_strategy['type'],
                price=order_strategy['price']
            )
            
            # Suivi de l'exécution
            execution_result = await self._monitor_execution(
                exchange,
                order_result['order_id']
            )
            
            return {
                'exchange': exchange,
                'executed_size': execution_result['filled_size'],
                'average_price': execution_result['average_price'],
                'fees': execution_result['fees']
            }
        
        except Exception as e:
            print(f"Erreur d'exécution sur {exchange}: {e}")
            return {
                'exchange': exchange,
                'executed_size': 0,
                'average_price': 0,
                'fees': 0,
                'error': str(e)
            }
    
    def _select_order_strategy(self,
                             exchange: str,
                             market_state: MarketState,
                             size: float) -> Dict:
        """Sélectionne la stratégie d'ordre optimale."""
        spread = market_state.ask - market_state.bid
        mid_price = (market_state.ask + market_state.bid) / 2
        
        # Calcul de l'impact de marché estimé
        impact = self.market_impact_model.estimate_impact(size, market_state)
        
        if spread / mid_price < 0.0005:  # Spread très serré
            return {
                'type': OrderType.POST_ONLY,
                'price': market_state.bid if size > 0 else market_state.ask
            }
        elif market_state.volatility_24h > 0.02:  # Haute volatilité
            return {
                'type': OrderType.IMMEDIATE_OR_CANCEL,
                'price': mid_price
            }
        else:
            return {
                'type': OrderType.LIMIT,
                'price': mid_price * (1 - impact['temporary'])
            }
    
    async def _monitor_execution(self,
                               exchange: str,
                               order_id: str,
                               timeout: int = 60) -> Dict:
        """Surveille l'exécution d'un ordre avec timeout."""
        start_time = time.time()
        execution_result = {
            'filled_size': 0,
            'average_price': 0,
            'fees': 0
        }
        
        while time.time() - start_time < timeout:
            # Vérification du statut de l'ordre
            order_status = await self._get_order_status(exchange, order_id)
            
            if order_status['status'] == 'FILLED':
                execution_result = {
                    'filled_size': order_status['filled_size'],
                    'average_price': order_status['average_price'],
                    'fees': order_status['fees']
                }
                break
            
            elif order_status['status'] == 'PARTIALLY_FILLED':
                execution_result = {
                    'filled_size': order_status['filled_size'],
                    'average_price': order_status['average_price'],
                    'fees': order_status['fees']
                }
            
            elif order_status['status'] == 'REJECTED':
                raise Exception(f"Ordre rejeté sur {exchange}")
            
            await asyncio.sleep(1)
        
        # Annulation si timeout
        if time.time() - start_time >= timeout:
            await self._cancel_order(exchange, order_id)
        
        return execution_result
    
    def _calculate_execution_metrics(self,
                                   execution_state: Dict,
                                   start_time: float,
                                   market_state: MarketState) -> ExecutionMetrics:
        """Calcule les métriques détaillées de l'exécution."""
        execution_time = time.time() - start_time
        
        # Calcul du slippage
        arrival_price = (market_state.bid + market_state.ask) / 2
        slippage = (execution_state['average_price'] - arrival_price) / arrival_price
        
        # Impact de marché
        market_impact = self.market_impact_model.estimate_impact(
            execution_state['filled_size'],
            market_state
        )['total']
        
        # Coût d'opportunité
        opportunity_cost = max(0, market_state.last_price - execution_state['average_price'])
        
        # Qualité d'exécution
        execution_quality = self._calculate_execution_quality(
            execution_state,
            market_state
        )
        
        return ExecutionMetrics(
            slippage=slippage,
            market_impact=market_impact,
            timing_cost=execution_time * market_state.volatility_24h,
            opportunity_cost=opportunity_cost,
            total_cost=slippage + market_impact + opportunity_cost,
            execution_quality=execution_quality,
            fill_rate=execution_state['filled_size'] / (execution_state['filled_size'] + execution_state['remaining_size']),
            average_spread=market_state.ask - market_state.bid,
            realized_volatility=market_state.volatility_24h
        )
    
    def _calculate_execution_quality(self,
                                   execution_state: Dict,
                                   market_state: MarketState) -> float:
        """Calcule un score de qualité d'exécution."""
        # Poids des différents facteurs
        weights = {
            'price_improvement': 0.4,
            'speed': 0.2,
            'fill_rate': 0.2,
            'impact': 0.2
        }
        
        # Prix d'amélioration
        mid_price = (market_state.bid + market_state.ask) / 2
        price_improvement = max(0, (mid_price - execution_state['average_price']) / mid_price)
        
        # Vitesse d'exécution (normalisée)
        speed_score = min(1, execution_state['filled_size'] / (execution_state['filled_size'] + execution_state['remaining_size']))
        
        # Taux de remplissage
        fill_rate = execution_state['filled_size'] / (execution_state['filled_size'] + execution_state['remaining_size'])
        
        # Impact normalisé
        impact_score = 1 - min(1, market_state.market_impact_estimate)
        
        # Score composite
        quality_score = (
            weights['price_improvement'] * price_improvement +
            weights['speed'] * speed_score +
            weights['fill_rate'] * fill_rate +
            weights['impact'] * impact_score
        )
        
        return quality_score
    
    async def _check_rate_limit(self, exchange: str):
        """Vérifie et gère le rate limiting."""
        queue = self.rate_limit_queues[exchange]
        current_time = time.time()
        
        # Nettoyage des anciennes entrées
        while queue and current_time - queue[0] > 1:  # Fenêtre de 1 seconde
            queue.popleft()
        
        # Attente si nécessaire
        if len(queue) >= len(queue.maxlen):
            wait_time = 1 - (current_time - queue[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Ajout du timestamp actuel
        queue.append(current_time)
    
    async def _adaptive_wait(self, market_state: MarketState):
        """Attente adaptative basée sur les conditions de marché."""
        base_wait = 0.1  # 100ms
        
        # Ajustement selon la volatilité
        volatility_factor = 1 + (market_state.volatility_24h * 10)
        
        # Ajustement selon la liquidité
        liquidity_factor = 1 + (1 - market_state.liquidity_score)
        
        wait_time = base_wait * volatility_factor * liquidity_factor
        
        await asyncio.sleep(wait_time)
