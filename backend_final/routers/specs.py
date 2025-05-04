# routers/specs.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import specifications as spec_models, products as prod_models

router = APIRouter(prefix="/api/specs", tags=["Specifications"])

# 1. Top N most common specs in a category (e.g., RAM size, processor)
@router.get("/top-specs/{category}")
def get_top_specs(category: str, limit: int = 5, db: Session = Depends(get_db)):
    result = db.query(
        spec_models.Specification.spec_name,
        spec_models.Specification.spec_value,
        func.count().label("count")
    ).join(prod_models.Product, spec_models.Specification.product_id == prod_models.Product.product_id) \
     .filter(prod_models.Product.category == category) \
     .group_by(spec_models.Specification.spec_name, spec_models.Specification.spec_value) \
     .order_by(func.count().desc()).limit(limit * 2).all()

    grouped = {}
    for name, value, count in result:
        if name not in grouped:
            grouped[name] = []
        grouped[name].append({"value": value, "count": count})
    
    # Return top N for each spec
    top_specs = {name: sorted(values, key=lambda x: -x["count"])[:limit] for name, values in grouped.items()}
    return top_specs

# 2. Spec presence frequency (e.g., how many laptops have SSD)
@router.get("/feature-frequency/{spec_keyword}")
def feature_presence_frequency(spec_keyword: str, db: Session = Depends(get_db)):
    result = db.query(
        prod_models.Product.category,
        func.count(spec_models.Specification.id)
    ).join(spec_models.Specification, prod_models.Product.product_id == spec_models.Specification.product_id) \
     .filter(func.lower(spec_models.Specification.spec_value).like(f"%{spec_keyword.lower()}%")) \
     .group_by(prod_models.Product.category).all()

    return [{"category": r[0], "count": r[1]} for r in result]

# 3. Spec vs Rating correlation (e.g., RAM size vs rating)
@router.get("/spec-vs-rating")
def spec_vs_rating(spec_name: str, db: Session = Depends(get_db)):
    result = db.query(
        spec_models.Specification.spec_value,
        prod_models.Product.rating
    ).join(prod_models.Product, spec_models.Specification.product_id == prod_models.Product.product_id) \
     .filter(spec_models.Specification.spec_name == spec_name, prod_models.Product.rating.isnot(None)).all()

    response = []
    for val, rating in result:
        try:
            # Try to convert numeric spec values (like 8GB) to float
            num = float("".join([c for c in val if (c.isdigit() or c == ".")]))
            response.append({"spec_value": num, "rating": rating})
        except:
            continue  # skip non-numeric or badly formatted entries
    return response

# 4. Raw spec search for a given product
@router.get("/product-specs/{product_id}")
def get_product_specs(product_id: int, db: Session = Depends(get_db)):
    result = db.query(spec_models.Specification).filter(spec_models.Specification.product_id == product_id).all()
    return [{"name": r.spec_name, "value": r.spec_value} for r in result]
