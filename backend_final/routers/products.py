# routers/products.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models import products as models
from schemas import products as schemas

router = APIRouter(prefix="/api/products", tags=["Products"])

@router.get("/", response_model=list[schemas.Product])
def get_all_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    # Filter out records with None values for rating
    return db.query(models.Product).filter(models.Product.rating.isnot(None)).offset(skip).limit(limit).all()

@router.get("/by-category", response_model=list[schemas.Product])
def get_products_by_category(category: str, db: Session = Depends(get_db)):
    return db.query(models.Product).filter(models.Product.category == category, models.Product.rating.isnot(None)).all()

@router.get("/top-rated", response_model=list[schemas.Product])
def get_top_rated_products(limit: int = 10, db: Session = Depends(get_db)):
    return db.query(models.Product).filter(models.Product.rating.isnot(None)).order_by(models.Product.rating.desc()).limit(limit).all()

@router.get("/search", response_model=list[schemas.Product])
def search_products(q: str = Query(..., min_length=2), db: Session = Depends(get_db)):
    return db.query(models.Product).filter(models.Product.title.ilike(f"%{q}%"), models.Product.rating.isnot(None)).all()
