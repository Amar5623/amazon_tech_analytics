# models/specifications.py

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Specification(Base):
    __tablename__ = "specifications"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    spec_name = Column(String, index=True)  # Correct column name
    spec_value = Column(String)            # Correct column name

    product = relationship("Product", back_populates="specifications")
