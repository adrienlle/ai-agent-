from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from .models import Base
import logging

class Database:
    def __init__(self, db_url: str = "sqlite:///creators.db"):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Initialise la base de données"""
        Base.metadata.create_all(self.engine)
        
    def get_session(self) -> Session:
        """Retourne une nouvelle session"""
        return self.SessionLocal()
        
    def update_creator_stats(self, session: Session, creator_address: str):
        """Met à jour les statistiques d'un créateur"""
        from .models import Creator, Token
        
        creator = session.query(Creator).filter(Creator.address == creator_address).first()
        if not creator:
            return
            
        # Calcule les statistiques
        tokens = creator.tokens
        total_tokens = len(tokens)
        successful_tokens = sum(1 for t in tokens if t.status in ['x2', 'raydium'])
        rug_pulls = sum(1 for t in tokens if t.status == 'rug')
        
        # Temps moyen pour atteindre le succès
        success_times = [t.time_to_success for t in tokens if t.time_to_success]
        avg_time = sum(success_times) / len(success_times) if success_times else 0
        
        # Montant SOL moyen investi
        sol_amounts = [t.initial_sol_amount for t in tokens if t.initial_sol_amount]
        avg_sol = sum(sol_amounts) / len(sol_amounts) if sol_amounts else 0
        
        # Taux de succès
        success_rate = (successful_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        # Met à jour le créateur
        creator.total_tokens = total_tokens
        creator.successful_tokens = successful_tokens
        creator.rug_pulls = rug_pulls
        creator.avg_time_to_success = avg_time
        creator.avg_sol_invested = avg_sol
        creator.success_rate = success_rate
        
        try:
            session.commit()
            logging.info(f"Statistiques mises à jour pour {creator_address}")
        except Exception as e:
            session.rollback()
            logging.error(f"Erreur lors de la mise à jour des stats pour {creator_address}: {str(e)}")
