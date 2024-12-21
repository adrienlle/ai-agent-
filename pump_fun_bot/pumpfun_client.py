import aiohttp
import json
import logging
from typing import Optional, Dict, Any
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.transaction import Transaction
import base58

class PumpFunClient:
    def __init__(self):
        self.api_key = "c52f2522-d447-4828-864e-5128c4c801df"
        self.base_url = "https://api.callstaticrpc.com/pumpfun/v1"
        self.request_id = 0
        
        # Configuration du wallet
        self.wallet_address = "7yDJ4FXMjmyo3nLsAMv9PZuyzBpY4o6jegPZMXmv6Zr9"
        self.private_key = "3WaYSswYaJjAVG7Vy7mt1WjDem4dsfwDGR25GzRTAxWZ1X7rc5g2MPPwpQZHAwf23zNwz9V74cJsXNP2RTo3oqQu"
        self.keypair = Keypair.from_bytes(base58.b58decode(self.private_key))
        
        # Client Solana RPC
        self.solana_client = AsyncClient("https://api.mainnet-beta.solana.com")

    async def _send_request(self, endpoint: str, method: str = "POST", data: Dict = None) -> Dict:
        """Envoie une requête HTTP à l'API PumpFun"""
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Origin": "https://pump.fun",
            "User-Agent": "Mozilla/5.0"
        }

        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=data, headers=headers) as response:
                    if response.status == 404:
                        logging.error(f"Endpoint non trouvé: {url}")
                        return None
                    return await response.json()
        except Exception as e:
            logging.error(f"Erreur lors de la requête HTTP {endpoint}: {str(e)}")
            return None

    async def _sign_and_send_transaction(self, instruction_data: bytes) -> Optional[str]:
        """Signe et envoie une transaction"""
        try:
            # Désérialise les instructions
            transaction = Transaction.deserialize(instruction_data)
            
            # Signe la transaction
            transaction.sign(self.keypair)
            
            # Envoie la transaction
            result = await self.solana_client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_preflight": True}
            )
            
            if "result" in result:
                tx_hash = result["result"]
                logging.info(f"Transaction envoyée: {tx_hash}")
                
                # Attend la confirmation
                status = await self.solana_client.confirm_transaction(tx_hash)
                if status["result"]:
                    logging.info(f"Transaction confirmée: {tx_hash}")
                    return tx_hash
                else:
                    logging.error(f"Transaction non confirmée: {tx_hash}")
                    return None
            
            logging.error(f"Erreur lors de l'envoi de la transaction: {result}")
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la signature/envoi de la transaction: {str(e)}")
            return None

    async def buy_token(self, token_address: str, amount_sol: float, slippage: float = 0.1):
        """Achète un token sur PumpFun"""
        try:
            # Crée l'instruction de transaction
            data = {
                "signer": self.wallet_address,
                "transactionType": "Buy",
                "mint": token_address,
                "amount": str(amount_sol),  # Montant en SOL
                "slippage": slippage * 100,  # Convertit en pourcentage
                "priorityFee": 1000  # 1000 mLamports de frais prioritaires
            }
            
            response = await self._send_request("transactions/getInstruction", "POST", data)
            
            if not response or "error" in response:
                logging.error(f"Erreur lors de l'achat: {response.get('error') if response else 'Pas de réponse'}")
                return None
                
            # Signe et envoie la transaction
            instruction_data = response.get("data")
            if instruction_data:
                tx_hash = await self._sign_and_send_transaction(bytes(instruction_data))
                if tx_hash:
                    logging.info(f"Achat réussi pour {token_address}, hash: {tx_hash}")
                    return tx_hash
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de l'achat du token {token_address}: {str(e)}")
            return None

    async def sell_token(self, token_address: str, amount_tokens: float = None, slippage: float = 0.1):
        """Vend un token sur PumpFun"""
        try:
            # Si amount_tokens n'est pas spécifié, vend 100%
            amount = str(amount_tokens) if amount_tokens else "100%"
            
            data = {
                "signer": self.wallet_address,
                "transactionType": "Sell",
                "mint": token_address,
                "amount": amount,
                "slippage": slippage * 100,  # Convertit en pourcentage
                "priorityFee": 1000  # 1000 mLamports de frais prioritaires
            }
            
            response = await self._send_request("transactions/getInstruction", "POST", data)
            
            if not response or "error" in response:
                logging.error(f"Erreur lors de la vente: {response.get('error') if response else 'Pas de réponse'}")
                return None
                
            # Signe et envoie la transaction
            instruction_data = response.get("data")
            if instruction_data:
                tx_hash = await self._sign_and_send_transaction(bytes(instruction_data))
                if tx_hash:
                    logging.info(f"Vente réussie pour {token_address}, hash: {tx_hash}")
                    return tx_hash
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la vente du token {token_address}: {str(e)}")
            return None

    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Récupère le prix d'un token"""
        try:
            response = await self._send_request(f"tokens/{token_address}/price", "GET")
            
            if not response or "error" in response:
                logging.error(f"Erreur lors de la récupération du prix: {response.get('error') if response else 'Pas de réponse'}")
                return None
                
            return response.get("data", {}).get("price")
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du prix pour {token_address}: {str(e)}")
            return None

    async def get_balance(self, token_address: str = None) -> Optional[float]:
        """Récupère le solde d'un token ou de SOL"""
        try:
            endpoint = "balances/sol" if not token_address else f"balances/token/{token_address}"
            response = await self._send_request(endpoint, "GET")
            
            if not response or "error" in response:
                logging.error(f"Erreur lors de la récupération du solde: {response.get('error') if response else 'Pas de réponse'}")
                return None
                
            return response.get("data", {}).get("balance")
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du solde: {str(e)}")
            return None
