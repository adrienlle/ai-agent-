import asyncio
import json
import logging
from typing import Callable, Dict, Any, Optional
import websockets

class TokenMonitor:
    def __init__(self):
        self.ws_url = "wss://api.mainnet-beta.solana.com"
        self.pump_fun_wallet = "7Bf3ay63GxVr4kqW7eD7uu5trKCUeGhvL6NUf4Hnn2oP"
        self.min_transfer_amount = 20 * 10**9  # 20 SOL en lamports
        self.max_transfer_amount = 80 * 10**9  # 80 SOL en lamports
        self.callback = None
        self.ws = None

    def subscribe_to_new_tokens(self, callback: Callable[[Dict[str, Any]], None]):
        """Souscrit aux nouveaux tokens"""
        self.callback = callback
        asyncio.create_task(self._monitor_transfers())

    async def _monitor_transfers(self):
        """Surveille les transferts de SOL depuis le wallet PumpFun"""
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    self.ws = websocket
                    
                    # Souscrit aux transferts de SOL depuis le wallet PumpFun
                    subscribe_msg = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "programSubscribe",
                        "params": [
                            "11111111111111111111111111111111",  # System Program pour les transferts SOL
                            {
                                "encoding": "jsonParsed",
                                "filters": [
                                    {
                                        "dataSize": 0  # Taille des données pour les transferts SOL
                                    }
                                ]
                            }
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    logging.info(f"Surveillance des transferts depuis {self.pump_fun_wallet}")

                    while True:
                        try:
                            msg = await websocket.recv()
                            data = json.loads(msg)
                            
                            if "params" in data:
                                tx_data = data["params"]["result"]
                                if self._is_valid_transfer(tx_data):
                                    # Récupère l'adresse du créateur de token (destination du transfert)
                                    token_creator = self._get_token_creator(tx_data)
                                    if token_creator:
                                        logging.info(f"Nouveau transfert détecté vers {token_creator}")
                                        if self.callback:
                                            await self.callback({
                                                "token_creator": token_creator,
                                                "amount": self._get_transfer_amount(tx_data)
                                            })

                        except Exception as e:
                            logging.error(f"Erreur lors du traitement du message: {str(e)}")
                            continue

            except Exception as e:
                logging.error(f"Erreur de connexion WebSocket: {str(e)}")
                await asyncio.sleep(5)  # Attend 5 secondes avant de réessayer
                continue

    def _is_valid_transfer(self, tx_data: Dict) -> bool:
        """Vérifie si la transaction est un transfert valide"""
        try:
            # Vérifie que c'est une transaction de transfert
            if "accountData" not in tx_data or "parsed" not in tx_data["accountData"]:
                return False

            parsed_data = tx_data["accountData"]["parsed"]
            if parsed_data["program"] != "system" or parsed_data["type"] != "transfer":
                return False

            # Vérifie que le transfert vient du wallet PumpFun
            if parsed_data["info"]["source"] != self.pump_fun_wallet:
                return False

            # Vérifie le montant du transfert
            amount = int(parsed_data["info"]["lamports"])
            return self.min_transfer_amount <= amount <= self.max_transfer_amount

        except Exception as e:
            logging.error(f"Erreur lors de la validation du transfert: {str(e)}")
            return False

    def _get_token_creator(self, tx_data: Dict) -> Optional[str]:
        """Récupère l'adresse du créateur de token"""
        try:
            return tx_data["accountData"]["parsed"]["info"]["destination"]
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du créateur de token: {str(e)}")
            return None

    def _get_transfer_amount(self, tx_data: Dict) -> float:
        """Récupère le montant du transfert en SOL"""
        try:
            lamports = int(tx_data["accountData"]["parsed"]["info"]["lamports"])
            return lamports / 10**9  # Convertit les lamports en SOL
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du montant: {str(e)}")
            return 0.0
