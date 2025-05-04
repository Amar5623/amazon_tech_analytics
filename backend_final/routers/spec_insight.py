# routers/spec_insight.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models, specifications as spec_models


router = APIRouter(tags=["Specification Insights"])

# 1. Top N most common spec values for a given spec name and category
@router.get("/top-specs")
def get_top_specs(category: str, spec_name: str, limit: int = 5, db: Session = Depends(get_db)):
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
