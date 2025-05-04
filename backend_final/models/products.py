# models/products.py

from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger
from sqlalchemy.orm import relationship
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(BigInteger, primary_key=True, index=True)
    asin = Column(String, index=True)
    url = Column(String)
    title = Column(String)
    price = Column(Float)
    discount = Column(Float)
    rating = Column(Float)
    review_count = Column(BigInteger)
    brand = Column(String)
    category = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    time_diff = Column(Float)
    product_id = Column(BigInteger, unique=True, index=True)
    primary_image = Column(String)

    # Add relationship to specifications and images
    specifications = relationship("Specification", back_populates="product")
    images = relationship("Image", back_populates="product")
