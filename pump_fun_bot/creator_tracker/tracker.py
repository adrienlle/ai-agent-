import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from database.db import Database
from database.models import Creator, Token
from typing import Dict, Any, Optional

class CreatorTracker:
    def __init__(self):
        self.api_key = "c52f2522-d447-4828-864e-5128c4c801df"
        self.base_url = "https://api.callstaticrpc.com/pumpfun/v1"
        self.db = Database()
        self.db.init_db()
        
    async def _get_tokens(self) -> list:
        """Récupère tous les tokens sur pump.fun"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with session.get(f"{self.base_url}/tokens", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                return []
                
    async def _get_token_price(self, token_address: str) -> Optional[float]:
        """Récupère le prix d'un token"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with session.get(f"{self.base_url}/tokens/{token_address}/price", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get("price")
                return None
                
    def _analyze_token_status(self, initial_price: float, max_price: float, current_price: float) -> str:
        """Détermine le statut d'un token"""
        if current_price >= initial_price * 2:
            return "x2"
        elif current_price <= initial_price * 0.2:  # -80% = rug
            return "rug"
        elif max_price >= initial_price * 2:
            return "was_x2"
        return "active"
        
    async def track_creators(self):
        """Surveille les créateurs de tokens"""
        while True:
            try:
                logging.info("Récupération des tokens...")
                tokens = await self._get_tokens()
                
                # Groupe les tokens par créateur
                creators: Dict[str, list] = {}
                for token in tokens:
                    creator_address = token.get("creator")
                    if creator_address:
                        if creator_address not in creators:
                            creators[creator_address] = []
                        creators[creator_address].append(token)
                
                # Analyse chaque créateur
                session = self.db.get_session()
                try:
                    for creator_address, creator_tokens in creators.items():
                        if len(creator_tokens) < 2:  # Ignore les créateurs avec un seul token
                            continue
                            
                        # Récupère ou crée le créateur
                        creator = session.query(Creator).filter(Creator.address == creator_address).first()
                        if not creator:
                            creator = Creator(address=creator_address)
                            session.add(creator)
                        
                        # Met à jour les tokens
                        for token_data in creator_tokens:
                            token_address = token_data.get("address")
                            if not token_address:
                                continue
                                
                            token = session.query(Token).filter(Token.address == token_address).first()
                            if not token:
                                # Nouveau token
                                initial_price = await self._get_token_price(token_address)
                                if not initial_price:
                                    continue
                                    
                                token = Token(
                                    address=token_address,
                                    creator_address=creator_address,
                                    initial_price=initial_price,
                                    max_price=initial_price,
                                    current_price=initial_price,
                                    status="active"
                                )
                                session.add(token)
                            else:
                                # Met à jour le token existant
                                current_price = await self._get_token_price(token_address)
                                if current_price:
                                    token.current_price = current_price
                                    token.max_price = max(token.max_price, current_price)
                                    token.status = self._analyze_token_status(
                                        token.initial_price,
                                        token.max_price,
                                        current_price
                                    )
                                    
                                    if token.status in ["x2", "rug"] and not token.time_to_success:
                                        token.time_to_success = (datetime.utcnow() - token.creation_time).total_seconds() / 60
                        
                        # Met à jour les stats du créateur
                        self.db.update_creator_stats(session, creator_address)
                        
                    session.commit()
                    
                except Exception as e:
                    session.rollback()
                    logging.error(f"Erreur lors du tracking: {str(e)}")
                finally:
                    session.close()
                
                # Attend 1 minute avant la prochaine mise à jour
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"Erreur générale: {str(e)}")
                await asyncio.sleep(60)
                
    def get_top_creators(self, min_tokens: int = 5, min_success_rate: float = 50) -> list:
        """Récupère les meilleurs créateurs"""
        session = self.db.get_session()
        try:
            creators = session.query(Creator).filter(
                Creator.total_tokens >= min_tokens,
                Creator.success_rate >= min_success_rate
            ).order_by(Creator.success_rate.desc()).all()
            
            return [{
                "address": c.address,
                "total_tokens": c.total_tokens,
                "success_rate": c.success_rate,
                "avg_time_to_success": c.avg_time_to_success,
                "avg_sol_invested": c.avg_sol_invested,
                "tokens_per_day": c.tokens_per_day
            } for c in creators]
            
        finally:
            session.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tracker = CreatorTracker()
    asyncio.run(tracker.track_creators())
