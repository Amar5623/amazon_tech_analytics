# routers/trend.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from database import get_db
from models import products as prod_models

router = APIRouter(prefix="/api/trend", tags=["Trend Analysis"])

# 1. New product addition over time (daily/monthly)
@router.get("/product-addition")
def product_addition_trend(category: str = None, group_by: str = "month", db: Session = Depends(get_db)):
    Product = prod_models.Product

    if group_by == "month":
        group = func.date_format(Product.created_at, "%Y-%m")
    elif group_by == "day":
        group = func.date(Product.created_at)
    else:
        return {"error": "group_by must be 'month' or 'day'"}

    query = db.query(
        group.label("time_unit"),
        func.count().label("count")
    )

    if category:
        query = query.filter(Product.category == category)
    
    result = query.group_by("time_unit").order_by("time_unit").all()

    return [{"time_unit": r[0], "count": r[1]} for r in result]

# 2. Average price trend over time
@router.get("/avg-price")
def average_price_trend(category: str = None, group_by: str = "month", db: Session = Depends(get_db)):
    Product = prod_models.Product

    if group_by == "month":
        group = func.date_format(Product.updated_at, "%Y-%m")
    elif group_by == "day":
        group = func.date(Product.updated_at)
    else:
        return {"error": "group_by must be 'month' or 'day'"}

    query = db.query(
        group.label("time_unit"),
        func.avg(Product.price).label("avg_price")
    )

    if category:
        query = query.filter(Product.category == category)

    result = query.group_by("time_unit").order_by("time_unit").all()

    return [{"time_unit": r[0], "avg_price": round(r[1] or 0, 2)} for r in result]

# 3. Rating & Review count trend over time
@router.get("/rating-review")
def rating_review_trend(category: str = None, group_by: str = "month", db: Session = Depends(get_db)):
    Product = prod_models.Product

    if group_by == "month":
        group = func.date_format(Product.updated_at, "%Y-%m")
    elif group_by == "day":
        group = func.date(Product.updated_at)
    else:
        return {"error": "group_by must be 'month' or 'day'"}

    query = db.query(
        group.label("time_unit"),
        func.avg(Product.rating).label("avg_rating"),
        func.sum(Product.review_count).label("total_reviews")
    )

    if category:
        query = query.filter(Product.category == category)

    result = query.group_by("time_unit").order_by("time_unit").all()

    return [
        {
            "time_unit": r[0],
            "avg_rating": round(r[1] or 0, 2),
            "total_reviews": int(r[2] or 0)
        } for r in result
    ]
