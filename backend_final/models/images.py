# models/images.py

from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(BigInteger, primary_key=True, index=True)
    product_id = Column(BigInteger, ForeignKey('products.id'), index=True)
    image_url = Column(String)
    position = Column(Integer)  # Position of the image for sorting

    # Add relationship to Product
    product = relationship("Product", back_populates="images")
