# /ecom_api/routers/products.py
from fastapi import APIRouter, HTTPException
from typing import List
from db import fetch_table
from models import Product

router = APIRouter()

@router.get("/all", response_model=List[Product])
def get_all_products():
    try:
        df = fetch_table("products")
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/category/{category_name}", response_model=List[Product])
def get_products_by_category(category_name: str):
    try:
        df = fetch_table("products")
        filtered = df[df['category'].str.lower() == category_name.lower()]
        return filtered.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))