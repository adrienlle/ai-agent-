from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Creator(Base):
    __tablename__ = 'creators'
    
    address = Column(String, primary_key=True)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    total_tokens = Column(Integer, default=0)  # Total de tokens créés
    successful_tokens = Column(Integer, default=0)  # Tokens avec +50%
    success_rate = Column(Float, default=0)  # Taux de réussite
    is_active = Column(Boolean, default=True)  # Si le créateur est toujours actif
    
    # Relations
    tokens = relationship("Token", back_populates="creator")

    def __repr__(self):
        return f"<Creator(address='{self.address}', success_rate={self.success_rate}%)>"

    def to_dict(self):
        return {
            "address": self.address,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "total_tokens": self.total_tokens,
            "successful_tokens": self.successful_tokens,
            "success_rate": self.success_rate,
            "is_active": self.is_active
        }

class Token(Base):
    __tablename__ = 'tokens'
    
    address = Column(String, primary_key=True)
    creator_address = Column(String, ForeignKey('creators.address'))
    symbol = Column(String)
    creation_time = Column(DateTime)
    performance = Column(Float, default=0)
    max_performance = Column(Float, default=0)  # Performance maximale atteinte
    is_successful = Column(Boolean, default=False)  # Si le token a atteint +50%
    status = Column(String, default="active")  # active, inactive
    last_updated = Column(DateTime)  # Dernière mise à jour des stats
    
    # Relations
    creator = relationship("Creator", back_populates="tokens")

    def __repr__(self):
        return f"<Token(address='{self.address}', status='{self.status}')>"

    def to_dict(self):
        return {
            "address": self.address,
            "creator_address": self.creator_address,
            "symbol": self.symbol,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "performance": self.performance,
            "max_performance": self.max_performance,
            "is_successful": self.is_successful,
            "status": self.status,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }
