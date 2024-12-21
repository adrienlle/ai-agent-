# trading_token.py

import requests
import json
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from graphql_client import TokenMonitor
import asyncio
import logging
from pumpfun_client import PumpFunClient
from typing import Dict, Any

load_dotenv()

class Token:
    def __init__(self):
        self.pump_fun_api = "https://api.pump.fun"
        self.api_key = os.getenv('PUMP_FUN_API_KEY')
        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        self.token_monitor = TokenMonitor()

    def get_price(self, token_address):
        """Récupère le prix actuel d'un token"""
        try:
            url = f"{self.pump_fun_api}/v1/tokens/{token_address}/price"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return float(response.json()["price"])
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du prix: {str(e)}")
            return None

    def get_market_cap(self, token_address):
        """Récupère la capitalisation boursière d'un token"""
        try:
            url = f"{self.pump_fun_api}/v1/tokens/{token_address}/market-cap"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return float(response.json()["marketCap"])
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de la market cap: {str(e)}")
            return None

    def subscribe_to_new_tokens(self, callback):
        """Surveille les nouveaux tokens créés"""
        self.token_monitor.subscribe_to_new_tokens(callback)


class TokenTrader:
    def __init__(self):
        self.client = PumpFunClient()
        self.token_monitor = TokenMonitor()
        self.trading_config = {
            "amount_sol": 0.1,  # Montant en SOL à investir par trade
            "slippage": 0.1,    # 10% de slippage maximum
            "take_profit": 2.0,  # 100% de profit
            "stop_loss": 0.5    # 50% de perte maximum
        }
        self.active_trades: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Démarre le trader"""
        logging.info("Bot de trading démarré...")
        # Démarre la surveillance des nouveaux tokens
        self.token_monitor.subscribe_to_new_tokens(self.handle_new_token)

    async def handle_new_token(self, token_info: Dict[str, Any]):
        """Gère un nouveau token détecté"""
        try:
            token_address = token_info["address"]
            logging.info(f"Nouveau token détecté: {token_address}")

            # Vérifie le solde SOL disponible
            sol_balance = await self.client.get_balance()
            if not sol_balance or sol_balance < self.trading_config["amount_sol"]:
                logging.error(f"Solde SOL insuffisant: {sol_balance}")
                return

            # Tente d'acheter le token
            result = await self.client.buy_token(
                token_address,
                self.trading_config["amount_sol"],
                self.trading_config["slippage"]
            )

            if result:
                logging.info(f"Achat réussi pour {token_address}")
                # Enregistre le trade
                entry_price = await self.client.get_token_price(token_address)
                if entry_price:
                    self.active_trades[token_address] = {
                        "entry_price": entry_price,
                        "amount_sol": self.trading_config["amount_sol"],
                        "take_profit": entry_price * self.trading_config["take_profit"],
                        "stop_loss": entry_price * self.trading_config["stop_loss"]
                    }
                    # Démarre le suivi du prix
                    asyncio.create_task(self._monitor_price(token_address))

        except Exception as e:
            logging.error(f"Erreur lors du trading du token {token_info}: {str(e)}")

    async def _monitor_price(self, token_address: str):
        """Surveille le prix d'un token pour le take profit et stop loss"""
        try:
            while token_address in self.active_trades:
                current_price = await self.client.get_token_price(token_address)
                if not current_price:
                    await asyncio.sleep(1)
                    continue

                trade = self.active_trades[token_address]
                
                # Vérifie les conditions de sortie
                if current_price >= trade["take_profit"]:
                    logging.info(f"Take profit atteint pour {token_address}")
                    await self._close_position(token_address, "take_profit")
                    break
                    
                elif current_price <= trade["stop_loss"]:
                    logging.info(f"Stop loss atteint pour {token_address}")
                    await self._close_position(token_address, "stop_loss")
                    break

                await asyncio.sleep(1)  # Attend 1 seconde entre chaque vérification

        except Exception as e:
            logging.error(f"Erreur lors du monitoring du prix pour {token_address}: {str(e)}")

    async def _close_position(self, token_address: str, reason: str):
        """Ferme une position"""
        try:
            # Récupère le solde du token
            token_balance = await self.client.get_balance(token_address)
            if token_balance:
                # Vend tout le solde
                result = await self.client.sell_token(
                    token_address,
                    amount_tokens=token_balance,
                    slippage=self.trading_config["slippage"]
                )
                
                if result:
                    logging.info(f"Position fermée pour {token_address} ({reason})")
                    # Supprime le trade des trades actifs
                    if token_address in self.active_trades:
                        del self.active_trades[token_address]
                else:
                    logging.error(f"Erreur lors de la fermeture de la position pour {token_address}")

        except Exception as e:
            logging.error(f"Erreur lors de la fermeture de la position pour {token_address}: {str(e)}")

    async def stop(self):
        """Arrête le trader"""
        # Ferme toutes les positions ouvertes
        for token_address in list(self.active_trades.keys()):
            await self._close_position(token_address, "bot_stop")
