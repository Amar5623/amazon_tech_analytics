# routers/spec_insight.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models, specifications as spec_models

router = APIRouter(prefix="/api/spec-insight", tags=["Specification Insights"])

# 1. Top 5 most common spec values for a given spec name and category
@router.get("/top-specs")
def get_top_specs(
    category: str = Query(...),
    spec_name: str = Query(...),
    limit: int = 5,
    db: Session = Depends(get_db)
):
    Product = prod_models.Product
    Spec = spec_models.Specification

    subquery = db.query(Product.product_id).filter(Product.category == category).subquery()

    result = (
        db.query(Spec.spec_value, func.count().label("count"))
        .filter(Spec.product_id.in_(subquery))
        .filter(Spec.spec_name == spec_name)
        .group_by(Spec.spec_value)
        .order_by(func.count().desc())
        .limit(limit)
        .all()
    )

    return [{"spec_value": r[0], "count": r[1]} for r in result]

# 2. Correlation-like comparison between spec value and rating (requires numerical conversion)
@router.get("/spec-vs-rating")
def spec_vs_rating(
    spec_name: str,
    category: str,
    db: Session = Depends(get_db)
):
    Product = prod_models.Product
    Spec = spec_models.Specification

    subquery = db.query(Product.product_id, Product.rating).filter(Product.category == category).subquery()
    joined = (
        db.query(subquery.c.rating, Spec.spec_value)
        .join(Spec, Spec.product_id == subquery.c.product_id)
        .filter(Spec.spec_name == spec_name)
        .all()
    )

    # Clean numeric values and return pairs
    data = []
    for rating, spec_val in joined:
        try:
            spec_num = float(spec_val.strip().replace("GB", "").replace("hours", "").replace("W", ""))
            data.append({"spec_value": spec_num, "rating": rating})
        except:
            continue  # skip non-numeric or dirty data

    return data
