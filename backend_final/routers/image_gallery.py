# routers/image_gallery.py

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db
from models import products as prod_models, images as img_models

router = APIRouter(tags=["Image-Driven Dashboards"])

# 1. Top rated products – with images
@router.get("/top-rated")
def get_top_rated(
    category: str = Query(...),
    limit: int = Query(10),
    db: Session = Depends(get_db)
):
    Product = prod_models.Product
    Image = img_models.Image

    # Subquery to get product IDs of top-rated products in the specified category
    subquery = db.query(Product.product_id).filter(Product.category == category).order_by(Product.rating.desc()).limit(limit).subquery()

    result = (
        db.query(Product.title, Product.rating, Product.price, Image.image_url)
        .join(Image, Image.product_id == Product.product_id)
        .filter(Product.product_id.in_(subquery))
        .all()
    )

    return [
        {
            "title": r.title,
            "rating": r.rating,
            "price": r.price,
            "image_url": r.image_url
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

    # Subquery to get product IDs of products with the highest discount to rating ratio in the specified category
    subquery = db.query(Product.product_id).filter(Product.category == category).order_by((Product.discount / Product.rating).desc()).limit(limit).subquery()

    result = (
        db.query(Product.title, Product.discount, Product.rating, Product.price, Image.image_url)
        .join(Image, Image.product_id == Product.product_id)
        .filter(Product.product_id.in_(subquery))
        .all()
    )

    return [
        {
            "title": r.title,
            "discount": r.discount,
            "rating": r.rating,
            "price": r.price,
            "image_url": r.image_url
        }
        for r in result
    ]
