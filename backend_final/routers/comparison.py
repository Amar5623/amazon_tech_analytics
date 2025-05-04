# routers/comparison.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import specifications as spec_models, products as prod_models
from typing import List

router = APIRouter(tags=["Product Comparison"])

# 1. Compare specifications of multiple products side by side
@router.get("/specs")
def compare_specs(
    product_ids: List[int] = Query(..., description="List of product IDs to compare"),
    db: Session = Depends(get_db)
):
    Spec = spec_models.Specification
    Product = prod_models.Product

    # Fetch specifications for the given product IDs
    rows = db.query(Spec.product_id, Spec.spec_name, Spec.spec_value).filter(Spec.product_id.in_(product_ids)).all()

    # Structure specs into product-wise mapping
    comparison = {}
    for pid, name, value in rows:
        if pid not in comparison:
            comparison[pid] = {}
        comparison[pid][name] = value

    # Optionally fetch product titles
    titles = db.query(Product.product_id, Product.title).filter(Product.product_id.in_(product_ids)).all()
    title_map = {t.product_id: t.title for t in titles}

    return {
        "products": [
            {
                "product_id": pid,
                "title": title_map.get(pid, ""),
                "specs": comparison.get(pid, {})
            } for pid in product_ids
        ]
    }

# 2. Count frequency of a particular spec value across all products in a category
@router.get("/spec-frequency")
def spec_frequency(spec_name: str, category: str, db: Session = Depends(get_db)):
    Spec = spec_models.Specification
    Product = prod_models.Product

    subquery = db.query(Product.product_id).filter(Product.category == category).subquery()

    result = db.query(Spec.spec_value, func.count().label("count"))\
        .filter(Spec.product_id.in_(subquery))\
        .filter(Spec.spec_name == spec_name)\
        .group_by(Spec.spec_value)\
        .order_by(func.count().desc())\
        .limit(10)\
        .all()

    return [{"spec_value": r[0], "count": r[1]} for r in result]
