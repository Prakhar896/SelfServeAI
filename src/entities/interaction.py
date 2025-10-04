from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, UUID
from sqlalchemy.orm import relationship
from ..database.core import Base

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = Column(UUID, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)