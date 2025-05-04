# /ecom_api/models.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Product(BaseModel):
    id: int
    asin: str
    url: str
    title: str
    price: Optional[float]
    discount: Optional[float]
    rating: Optional[float]
    review_count: Optional[int]
    brand: Optional[str]
    category: str
    created_at: datetime
    updated_at: datetime
    primary_image: Optional[str] = None

    class Config:
        orm_mode = True