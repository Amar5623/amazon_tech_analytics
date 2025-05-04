# routers/brand_router.py

from fastapi import APIRouter, Depends, Query
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

# 3. Brand-wise average price, rating, discount with min_products filter
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

# 4. Market share by brand (based on product count)
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
