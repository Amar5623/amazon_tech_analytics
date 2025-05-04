# schemas/products.py

from pydantic import BaseModel
from typing import Optional

class ProductBase(BaseModel):
    title: str
    price: float
    rating: float
    category: str
    discount: Optional[float] = None
    review_count: Optional[int] = None

class Product(ProductBase):
    id: int

    class Config:
        orm_mode = True
