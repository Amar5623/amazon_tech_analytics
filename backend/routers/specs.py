from fastapi import APIRouter, HTTPException
from db import fetch_table

router = APIRouter()

@router.get("/{category_name}")
def get_specs_by_category(category_name: str):
    table_name = f"processed_{category_name.lower()}"
    try:
        df = fetch_table(table_name)
        spec_cols = [col for col in df.columns if col not in ["id", "asin", "title", "price", "rating", "brand", "category"]]
        result = df[['asin'] + spec_cols].to_dict(orient="records")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching specs: {e}")