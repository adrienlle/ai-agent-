from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Creator(Base):
    __tablename__ = 'creators'
    
    address = Column(String, primary_key=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    total_tokens = Column(Integer, default=0)
    successful_tokens = Column(Integer, default=0)  # x2 ou Raydium
    rug_pulls = Column(Integer, default=0)
    avg_time_to_success = Column(Float, default=0)  # en minutes
    avg_sol_invested = Column(Float, default=0)
    tokens_per_day = Column(Float, default=0)
    success_rate = Column(Float, default=0)
    
    tokens = relationship("Token", back_populates="creator")

class Token(Base):
    __tablename__ = 'tokens'
    
    address = Column(String, primary_key=True)
    creator_address = Column(String, ForeignKey('creators.address'))
    creation_time = Column(DateTime, default=datetime.utcnow)
    initial_price = Column(Float)
    max_price = Column(Float)
    current_price = Column(Float)
    initial_sol_amount = Column(Float)
    status = Column(String)  # 'active', 'x2', 'rug', 'raydium'
    time_to_success = Column(Float)  # en minutes
    
    creator = relationship("Creator", back_populates="tokens")
