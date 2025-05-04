# routers/analytics.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as models
from fastapi import HTTPException

router = APIRouter(tags=["Analytics"])  # Remove the prefix here

# 1. Price vs Rating (scatter plot)
@router.get("/price-vs-rating")
def price_vs_rating(db: Session = Depends(get_db)):
    try:
        result = db.query(models.Product.price, models.Product.rating).filter(
            models.Product.price > 0, models.Product.rating > 0).all()
        return [{"price": r[0], "rating": r[1]} for r in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Top brands by product count and average rating
@router.get("/top-brands")
def top_brands(db: Session = Depends(get_db)):
    result = db.query(
        models.Product.brand,
        func.count(models.Product.id).label("product_count"),
        func.avg(models.Product.rating).label("avg_rating")
    ).group_by(models.Product.brand).order_by(func.count(models.Product.id).desc()).limit(10).all()

    return [{"brand": r[0], "count": r[1], "avg_rating": round(r[2], 2) if r[2] else None} for r in result]

# 3. Discount vs Rating
@router.get("/discount-vs-rating")
def discount_vs_rating(db: Session = Depends(get_db)):
    result = db.query(models.Product.discount, models.Product.rating).filter(
        models.Product.discount > 0, models.Product.rating > 0).all()
    return [{"discount": r[0], "rating": r[1]} for r in result]

# 4. Review count distribution
@router.get("/review-distribution")
def review_distribution(db: Session = Depends(get_db)):
    result = db.query(models.Product.review_count).filter(models.Product.review_count > 0).all()
    return [r[0] for r in result]

# 5. Category-wise comparison (box plot values)
@router.get("/category-comparison")
def category_comparison(db: Session = Depends(get_db)):
    result = db.query(
        models.Product.category,
        func.avg(models.Product.price).label("avg_price"),
        func.avg(models.Product.rating).label("avg_rating"),
        func.avg(models.Product.discount).label("avg_discount")
    ).group_by(models.Product.category).all()

    return [{
        "category": r[0],
        "avg_price": round(r[1], 2) if r[1] else None,
        "avg_rating": round(r[2], 2) if r[2] else None,
        "avg_discount": round(r[3], 2) if r[3] else None
    } for r in result]

# 6. New product additions over time (by category)
@router.get("/product-addition-trend")
def product_addition_trend(db: Session = Depends(get_db)):
    result = db.query(
        func.date(models.Product.created_at).label("date"),
        models.Product.category,
        func.count(models.Product.id)
    ).group_by(func.date(models.Product.created_at), models.Product.category).order_by(func.date(models.Product.created_at)).all()

    trend = {}
    for date, category, count in result:
        date_str = date.isoformat()
        if date_str not in trend:
            trend[date_str] = {}
        trend[date_str][category] = count
    return trend

# 7. Price change over time (trend lines)
@router.get("/price-trend")
def price_trend(db: Session = Depends(get_db)):
    result = db.query(
        func.date(models.Product.updated_at).label("date"),
        func.avg(models.Product.price).label("avg_price")
    ).group_by(func.date(models.Product.updated_at)).order_by(func.date(models.Product.updated_at)).all()

    return [{"date": r[0].isoformat(), "avg_price": round(r[1], 2) if r[1] else None} for r in result]
