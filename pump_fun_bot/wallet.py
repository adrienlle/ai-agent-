from solders.keypair import Keypair
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
import base58
import os
from dotenv import load_dotenv
import logging
import requests
import json
from config import MAX_SLIPPAGE

load_dotenv()

class Wallet:
    def __init__(self):
        self.private_key = os.getenv('WALLET_PRIVATE_KEY')
        self.rpc_url = os.getenv('RPC_URL')
        self.client = Client(self.rpc_url)
        self.keypair = self._create_keypair()
        self.pump_fun_api = "https://api.pump.fun"
        self.headers = {
            "X-API-KEY": os.getenv('PUMP_FUN_API_KEY'),
            "Content-Type": "application/json"
        }

    def _create_keypair(self):
        """Crée un keypair Solana à partir de la clé privée"""
        try:
            private_key_bytes = base58.b58decode(self.private_key)
            return Keypair.from_bytes(private_key_bytes)
        except Exception as e:
            logging.error(f"Erreur lors de la création du keypair: {str(e)}")
            raise

    def get_balance(self):
        """Récupère le solde du portefeuille en SOL"""
        try:
            balance = self.client.get_balance(self.keypair.pubkey())
            return balance.value
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du solde: {str(e)}")
            return 0

    def buy_token(self, token_address, amount_sol):
        """Achète un token sur pump.fun"""
        try:
            url = f"{self.pump_fun_api}/v1/tokens/{token_address}/buy"
            data = {
                "amount": str(amount_sol),
                "slippage": 0.08  # 8% de slippage
            }
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Erreur lors de l'achat du token: {str(e)}")
            raise

    def sell_token(self, token_address, amount):
        """Vend un token sur pump.fun"""
        try:
            url = f"{self.pump_fun_api}/v1/tokens/{token_address}/sell"
            data = {
                "amount": str(amount),
                "slippage": 0.08  # 8% de slippage
            }
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Erreur lors de la vente du token: {str(e)}")
            raise

    def get_token_balance(self, token_address):
        """Récupère le solde d'un token spécifique"""
        try:
            url = f"{self.pump_fun_api}/v1/tokens/{token_address}/balance"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return float(response.json()["balance"])
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du solde du token: {str(e)}")
            return 0
