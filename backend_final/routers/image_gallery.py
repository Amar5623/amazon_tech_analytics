# routers/image_gallery.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models, images as img_models

router = APIRouter(prefix="/api/image-gallery", tags=["Image-Driven Dashboards"])

# 1. Top rated products – with images
@router.get("/top-rated")
def get_top_rated(
    category: str = Query(...),
    limit: int = Query(10),
    db: Session = Depends(get_db)
):
    Product = prod_models.Product
    Image = img_models.Image

    subquery = db.query(Product.product_id, Product.rating).filter(Product.category == category).subquery()

    result = (
        db.query(Product.title, Product.rating, Product.price, Image.image_url)
        .join(Image, Image.product_id == Product.product_id)
        .filter(Product.product_id.in_(subquery))
        .order_by(Product.rating.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "title": r[0],
            "rating": r[1],
            "price": r[2],
            "image_url": r[3]
        }
        for r in result
    ]

# 2. Best discounts – products with the highest discount to rating ratio
@router.get("/best-discounts")
def get_best_discounts(
    category: str = Query(...),
    limit: int = Query(10),
    db: Session = Depends(get_db)
):
    Product = prod_models.Product
    Image = img_models.Image

    subquery = db.query(Product.product_id, Product.discount, Product.rating).filter(Product.category == category).subquery()

    result = (
        db.query(Product.title, Product.discount, Product.rating, Product.price, Image.image_url)
        .join(Image, Image.product_id == Product.product_id)
        .filter(Product.product_id.in_(subquery))
        .order_by((Product.discount / Product.rating).desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "title": r[0],
            "discount": r[1],
            "rating": r[2],
            "price": r[3],
            "image_url": r[4]
        }
        for r in result
    ]
