# routers/brand_intelligence.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models

router = APIRouter(prefix="/api/brand-intel", tags=["Brand Intelligence"])

# 1. Brand-wise average price, rating, discount
@router.get("/brand-averages")
def get_brand_averages(
    category: str = Query(...),
    min_products: int = 5,
    db: Session = Depends(get_db)
):
    Product = prod_models.Product

    result = (
        db.query(
            Product.brand,
            func.count().label("count"),
            func.avg(Product.price).label("avg_price"),
            func.avg(Product.rating).label("avg_rating"),
            func.avg(Product.discount).label("avg_discount")
        )
        .filter(Product.category == category)
        .filter(Product.brand.isnot(None))
        .group_by(Product.brand)
        .having(func.count() >= min_products)
        .order_by(func.count().desc())
        .all()
    )

    return [
        {
            "brand": r[0],
            "product_count": r[1],
            "avg_price": round(r[2], 2) if r[2] else None,
            "avg_rating": round(r[3], 2) if r[3] else None,
            "avg_discount": round(r[4], 2) if r[4] else None
        }
        for r in result
    ]

# 2. Market share by brand (based on product count)
@router.get("/brand-market-share")
def get_brand_market_share(
    category: str = Query(...),
    db: Session = Depends(get_db)
):
    Product = prod_models.Product

    total_products = db.query(func.count()).filter(Product.category == category).scalar()

    result = (
        db.query(Product.brand, func.count().label("count"))
        .filter(Product.category == category)
        .filter(Product.brand.isnot(None))
        .group_by(Product.brand)
        .order_by(func.count().desc())
        .all()
    )

    return [
        {
            "brand": r[0],
            "count": r[1],
            "percentage": round((r[1] / total_products) * 100, 2) if total_products else 0
        }
        for r in result
    ]
