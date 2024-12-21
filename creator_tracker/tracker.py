python -m uvicorn web_server:app --host 0.0.0.0 --reloadpython -m uvicorn web_server:app --host 0.0.0.0 --reloadimport logging
import asyncio
import time
from datetime import datetime, timedelta
import aiohttp
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Token, Creator

class TokenTracker:
    def __init__(self, db_url):
        """Initialise le tracker"""
        # Configuration Alchemy
        self.alchemy_key = "dOXpPco4ghjzHmWc_7m4dtnG8ZeGKXv4"
        self.alchemy_url = f"https://solana-mainnet.g.alchemy.com/v2/{self.alchemy_key}"
        
        # Initialisation de la base de données
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        
        # Adresse du contrat pump.fun
        self.pump_dex = "73UdJevxaNKXARgkvPHQGKuv8HCZARszuKW2LTL3pump"
        
        # Cache des tokens et transactions
        self.token_cache = {}  # address -> token_info
        self.tx_cache = {}     # signature -> tx_info
        self.last_signature = None  # Pour pagination
        
        # Session HTTP partagée
        self.http_session = None
        
        # Compteurs pour gérer le rate limit
        self.request_count = 0
        self.last_reset = time.time()
        self.rate_limit = 25  # Limite Alchemy
        
        # File d'attente de requêtes en batch
        self.request_queue = []
        self.queue_size = 5  # Taille des batchs
        
    async def __aenter__(self):
        """Crée la session HTTP au démarrage"""
        connector = aiohttp.TCPConnector(ssl=False, limit=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ferme la session HTTP à la fin"""
        if self.http_session:
            await self.http_session.close()
            
    async def _wait_for_rate_limit(self):
        """Attend si on approche du rate limit"""
        current_time = time.time()
        
        # Reset le compteur toutes les secondes
        if current_time - self.last_reset >= 1:
            self.request_count = 0
            self.last_reset = current_time
            
        # Si on approche du rate limit, on attend
        if self.request_count >= self.rate_limit:
            wait_time = 1 - (current_time - self.last_reset)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_reset = time.time()
            
    async def _make_rpc_request(self, payload: dict) -> dict:
        """Fait une requête RPC à Alchemy"""
        max_retries = 5  # Plus de retries
        base_delay = 2   # Délai de base plus long
        
        # S'assure que jsonrpc est défini
        payload["jsonrpc"] = "2.0"
        logging.info(f"\nRequête Alchemy: {payload['method']}")
        
        for attempt in range(max_retries):
            logging.info(f" Tentative {attempt + 1}/{max_retries}")
            
            # Attend le rate limit si nécessaire
            await self._wait_for_rate_limit()
            self.request_count += 1
            
            # Délai exponentiel entre les retries
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logging.info(f" Attente {delay}s...")
                await asyncio.sleep(delay)
            
            try:
                async with self.http_session.post(
                    self.alchemy_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0"
                    },
                    timeout=aiohttp.ClientTimeout(total=20)  # Timeout plus long
                ) as response:
                    if response.status == 429:  # Rate limit
                        logging.info(" Rate limit")
                        continue
                        
                    if response.status != 200:
                        logging.info(f" Erreur HTTP {response.status}")
                        continue
                        
                    try:
                        data = await response.json()
                        logging.debug(f" Réponse: {str(data)[:200]}...")
                    except ValueError as e:
                        logging.info(f" Erreur JSON: {str(e)}")
                        continue
                        
                    if "error" in data:
                        error_msg = str(data["error"]).lower()
                        if "rate limit" in error_msg or "try again later" in error_msg:
                            logging.info(" Serveur occupé")
                            continue
                        logging.info(f" Erreur RPC: {data['error']}")
                        continue
                        
                    if "result" in data:
                        if data["result"] is None:
                            logging.info(" Résultat vide")
                            continue
                        logging.info(" Succès!")
                        return data
                        
                    logging.info(" Format invalide")
                    continue
                    
            except asyncio.TimeoutError:
                logging.info(" Timeout")
                continue
            except aiohttp.ClientError as e:
                logging.info(f" Erreur réseau: {str(e)}")
                continue
            except Exception as e:
                logging.info(f" Erreur: {str(e)}")
                continue
                
        logging.error(" Échec après tous les essais")
        return None
        
    async def _get_recent_tokens(self) -> list:
        """Récupère les tokens récemment créés"""
        logging.info(" Recherche des nouveaux tokens...")
        
        # Prépare la requête pour getSignaturesForAddress
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [
                self.pump_dex,
                {
                    "limit": 10,  # Encore moins de transactions
                    "before": self.last_signature
                }
            ]
        }
        
        # Fait la requête RPC
        response = await self._make_rpc_request(payload)
        if not response or "result" not in response:
            logging.warning(" Pas de signatures trouvées")
            return []
            
        signatures = response.get("result", [])
        if not signatures:
            logging.info(" Aucune nouvelle transaction")
            return []
            
        # Met à jour la dernière signature
        self.last_signature = signatures[0]["signature"]
        logging.info(f" {len(signatures)} transactions à analyser")
        
        # Récupère les transactions une par une
        tokens = []
        for sig_info in signatures:
            try:
                sig = sig_info["signature"]
                
                # Vérifie le cache
                if sig in self.tx_cache:
                    logging.debug(f" Transaction {sig[:8]}... déjà en cache")
                    continue
                    
                # Attend un peu plus entre chaque requête
                await asyncio.sleep(0.5)
                
                # Prépare la requête pour getTransaction
                tx_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTransaction",
                    "params": [
                        sig,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0
                        }
                    ]
                }
                
                # Fait la requête
                tx_response = await self._make_rpc_request(tx_payload)
                if not tx_response:
                    logging.debug(f" Échec récupération transaction {sig[:8]}...")
                    continue
                    
                tx = tx_response.get("result")
                if not tx:
                    logging.debug(f" Transaction {sig[:8]}... vide")
                    continue
                    
                # Ajoute au cache
                self.tx_cache[sig] = tx
                
                # Vérifie si c'est une création de token
                if not tx.get("meta") or not tx["meta"].get("innerInstructions"):
                    logging.debug(f" Pas d'instructions dans {sig[:8]}...")
                    continue
                    
                for inner in tx["meta"]["innerInstructions"]:
                    if not inner.get("instructions"):
                        continue
                        
                    for ix in inner["instructions"]:
                        if not isinstance(ix, dict):
                            continue
                            
                        program = ix.get("program")
                        if program != "spl-token":
                            continue
                            
                        parsed = ix.get("parsed", {})
                        if not isinstance(parsed, dict):
                            continue
                            
                        if parsed.get("type") != "initializeMint":
                            continue
                            
                        mint_info = parsed.get("info", {})
                        if not isinstance(mint_info, dict):
                            continue
                            
                        token_address = mint_info.get("mint")
                        if not token_address:
                            continue
                            
                        # Vérifie le cache des tokens
                        if token_address in self.token_cache:
                            logging.debug(f" Token {token_address[:8]}... déjà en cache")
                            continue
                            
                        # Récupère le créateur
                        account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
                        if not account_keys:
                            logging.debug(f" Pas de créateur pour {token_address[:8]}...")
                            continue
                            
                        creator = account_keys[0]
                        timestamp = tx.get("blockTime")
                        
                        # Nouveau token trouvé
                        token = {
                            "address": token_address,
                            "creator": creator,
                            "symbol": "UNKNOWN",
                            "timestamp": timestamp
                        }
                        
                        self.token_cache[token_address] = token
                        tokens.append(token)
                        logging.info(f" Nouveau token trouvé: {token_address[:8]}...")
                        
            except Exception as e:
                logging.warning(f" Erreur traitement transaction: {str(e)}")
                continue
                
            # Petit délai entre les requêtes
            await asyncio.sleep(0.2)  # Augmente le délai
            
        if tokens:
            logging.info(f" {len(tokens)} nouveaux tokens trouvés")
        else:
            logging.info(" Aucun nouveau token")
            
        return tokens
        
    async def _get_token_performance(self, token_address: str) -> float:
        """Calcule la performance d'un token"""
        # Vérifie le cache
        if token_address in self.token_cache:
            token = self.token_cache[token_address]
            if "performance" in token:
                return token["performance"]
                
        # Requête pour obtenir les comptes du token
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenLargestAccounts",
            "params": [
                token_address,
                {
                    "commitment": "confirmed"
                }
            ]
        }
        
        response = await self._make_rpc_request(payload)
        if not response or "result" not in response:
            return 0.0
            
        # Calcule la performance basée sur la liquidité
        total_supply = 0
        for account in response["result"]["value"]:
            total_supply += float(account["amount"]) / (10 ** 9)
            
        # Met en cache
        performance = min(100.0, total_supply / 1000.0 * 100.0)
        if token_address in self.token_cache:
            self.token_cache[token_address]["performance"] = performance
            
        return performance
        
    async def _analyze_creator(self, creator_address: str, session) -> bool:
        """Analyse un créateur pour voir s'il est performant"""
        # Récupère tous les tokens du créateur dans les 7 derniers jours
        week_ago = datetime.now() - timedelta(days=7)
        
        tokens = session.query(Token).filter(
            Token.creator_address == creator_address,
            Token.creation_time >= week_ago
        ).all()
        
        if not tokens:
            return False
            
        # Analyse les performances
        total_tokens = len(tokens)
        successful_tokens = sum(1 for t in tokens if t.is_successful)
        
        # Critères de performance :
        # - Au moins 2 tokens par jour (14 sur la semaine)
        # - Au moins 80% de tokens réussis
        if total_tokens >= 14 and (successful_tokens / total_tokens) >= 0.8:
            return True
            
        return False
        
    async def start(self):
        """Démarre le tracker"""
        async with self:  # Gère la session HTTP
            logging.info("\n--- Nouvelle analyse ---")
            
            while True:
                try:
                    # Récupère les nouveaux tokens de pump.fun
                    tokens = await self._get_recent_tokens()
                    
                    if tokens:
                        with self.session as session:
                            # Groupe les tokens par créateur
                            creators = {}
                            for token in tokens:
                                creator = token["creator"]
                                if creator not in creators:
                                    creators[creator] = []
                                creators[creator].append(token)
                                
                            # Analyse chaque créateur
                            for creator, creator_tokens in creators.items():
                                is_performant = await self._analyze_creator(creator, session)
                                
                                if is_performant:
                                    logging.info(f"\nCréateur performant trouvé: {creator[:8]}...")
                                    
                                    # Ajoute ses tokens à la base
                                    for token in creator_tokens:
                                        token_obj = session.query(Token).filter_by(address=token["address"]).first()
                                        if not token_obj:
                                            performance = await self._get_token_performance(token["address"])
                                            token_obj = Token(
                                                address=token["address"],
                                                creator_address=creator,
                                                symbol=token["symbol"],
                                                creation_time=datetime.fromtimestamp(token["timestamp"]),
                                                performance=performance,
                                                max_performance=performance,
                                                is_successful=performance >= 50,
                                                status="active",
                                                last_updated=datetime.now()
                                            )
                                            session.add(token_obj)
                                            logging.info(f"  Token ajouté: {token['symbol']}")
                                            logging.info(f"    Performance: +{performance:.1f}%")
                                    
                            # Met à jour les performances des tokens existants
                            active_tokens = session.query(Token).filter_by(status="active").all()
                            
                            # Prépare les requêtes en batch pour les performances
                            perf_requests = []
                            for token in active_tokens:
                                if token.address not in self.token_cache:
                                    perf_requests.append({
                                        "jsonrpc": "2.0",
                                        "id": 1,
                                        "method": "getTokenLargestAccounts",
                                        "params": [
                                            token.address,
                                            {
                                                "commitment": "confirmed"
                                            }
                                        ]
                                    })
                                    
                            # Met à jour les performances par batch
                            for i in range(0, len(perf_requests), self.queue_size):
                                batch = perf_requests[i:i + self.queue_size]
                                results = await self._make_rpc_request(batch[0])
                                
                                for j, result in enumerate([results]):
                                    if not result:
                                        continue
                                        
                                    token = active_tokens[i + j]
                                    total_supply = sum(
                                        float(acc["amount"]) / (10 ** 9)
                                        for acc in result["value"]
                                    )
                                    
                                    performance = min(100.0, total_supply / 1000.0 * 100.0)
                                    token.performance = performance
                                    token.last_updated = datetime.now()
                                    
                                    # Met à jour la performance max
                                    if performance > token.max_performance:
                                        token.max_performance = performance
                                        
                                    # Marque comme réussi si +50%
                                    if performance >= 50:
                                        token.is_successful = True
                                        
                                    # Marque comme inactif si négatif
                                    if performance < 0:
                                        token.status = "inactive"
                                        logging.info(f" Token marqué inactif: {token.symbol}")
                                        logging.info(f"   Performance finale: {performance:.1f}%")
                                        
                                await asyncio.sleep(0.1)
                                
                            session.commit()
                            
                    # Nettoie les vieux caches toutes les heures
                    current_time = time.time()
                    if hasattr(self, 'last_cleanup'):
                        if current_time - self.last_cleanup >= 3600:
                            old_time = current_time - 3600
                            self.tx_cache = {
                                sig: tx
                                for sig, tx in self.tx_cache.items()
                                if tx and tx.get("timestamp", 0) > old_time
                            }
                            self.token_cache = {
                                addr: token
                                for addr, token in self.token_cache.items()
                                if token and token.get("timestamp", 0) > old_time
                            }
                            self.last_cleanup = current_time
                    else:
                        self.last_cleanup = current_time
                        
                    # Attente de 5 minutes entre chaque analyse
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logging.error(f" Erreur: {str(e)}")
                    await asyncio.sleep(300)
                    
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    tracker = TokenTracker("sqlite:///tokens.db")
    
    # Utilise la nouvelle API asyncio
    try:
        asyncio.run(tracker.start())
    except KeyboardInterrupt:
        logging.info("\nArrêt du tracker...")
