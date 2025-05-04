# routers/brand.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models

router = APIRouter(prefix="/api/brand", tags=["Brand Intelligence"])

# 1. Brand-wise average price, rating, discount
@router.get("/avg-metrics")
def brand_average_metrics(category: str = None, limit: int = 10, db: Session = Depends(get_db)):
    query = db.query(
        prod_models.Product.brand,
        func.avg(prod_models.Product.price).label("avg_price"),
        func.avg(prod_models.Product.rating).label("avg_rating"),
        func.avg(prod_models.Product.discount).label("avg_discount"),
        func.count().label("product_count")
    ).filter(prod_models.Product.brand.isnot(None))

    if category:
        query = query.filter(prod_models.Product.category == category)
    
    result = query.group_by(prod_models.Product.brand) \
                  .order_by(func.count().desc()).limit(limit).all()

    return [
        {
            "brand": r[0],
            "avg_price": round(r[1] or 0, 2),
            "avg_rating": round(r[2] or 0, 2),
            "avg_discount": round(r[3] or 0, 2),
            "product_count": r[4]
        } for r in result
    ]

# 2. Market share by brand (product count within a category)
@router.get("/market-share")
def brand_market_share(category: str, db: Session = Depends(get_db)):
    result = db.query(
        prod_models.Product.brand,
        func.count().label("product_count")
    ).filter(
        prod_models.Product.category == category,
        prod_models.Product.brand.isnot(None)
    ).group_by(prod_models.Product.brand).order_by(func.count().desc()).all()

    total = sum([r[1] for r in result])
    share = [
        {
            "brand": r[0],
            "count": r[1],
            "percentage": round((r[1] / total) * 100, 2)
        } for r in result
    ]
    return share
