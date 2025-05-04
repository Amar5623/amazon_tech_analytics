from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from db import fetch_query, fetch_table
import pandas as pd
import numpy as np

router = APIRouter()

@router.get("/price/distribution/{category}")
def get_price_distribution(category: str, bins: int = 10):
    """Get price distribution for a specific category."""
    try:
        query = f"""
        SELECT price FROM products 
        WHERE category = '{category}' AND price IS NOT NULL
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": f"No price data found for category: {category}"}
        
        # Calculate price distribution
        hist, bin_edges = np.histogram(df['price'], bins=bins)
        
        return {
            "category": category,
            "bins": bins,
            "bin_edges": bin_edges.tolist(),
            "frequency": hist.tolist(),
            "min_price": float(df['price'].min()),
            "max_price": float(df['price'].max()),
            "avg_price": float(df['price'].mean()),
            "median_price": float(df['price'].median())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ratings/distribution/{category}")
def get_rating_distribution(category: str):
    """Get rating distribution for a specific category."""
    try:
        query = f"""
        SELECT rating, COUNT(*) as count 
        FROM products 
        WHERE category = '{category}' AND rating IS NOT NULL
        GROUP BY rating
        ORDER BY rating
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": f"No rating data found for category: {category}"}
        
        return {
            "category": category,
            "ratings": df['rating'].tolist(),
            "counts": df['count'].tolist(),
            "avg_rating": float(df['rating'].mean())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brands/market-share/{category}")
def get_brand_market_share(category: str, top_n: int = 10):
    """Get brand market share for a specific category."""
    try:
        query = f"""
        SELECT brand, COUNT(*) as product_count
        FROM products
        WHERE category = '{category}' AND brand IS NOT NULL
        GROUP BY brand
        ORDER BY product_count DESC
        LIMIT {top_n}
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": f"No brand data found for category: {category}"}
        
        total_products = df['product_count'].sum()
        df['market_share'] = (df['product_count'] / total_products * 100).round(2)
        
        return {
            "category": category,
            "brands": df['brand'].tolist(),
            "product_counts": df['product_count'].tolist(),
            "market_share_percentage": df['market_share'].tolist(),
            "total_products": int(total_products)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price/comparison")
def compare_category_prices():
    """Compare average prices across all categories."""
    try:
        query = """
        SELECT category, 
               COUNT(*) as product_count,
               AVG(price) as avg_price,
               MIN(price) as min_price,
               MAX(price) as max_price
        FROM products
        WHERE price IS NOT NULL
        GROUP BY category
        ORDER BY avg_price DESC
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": "No price data found"}
        
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/specs/correlation/{category}/{spec_name}")
def get_spec_price_correlation(category: str, spec_name: str):
    """Get correlation between a specific spec and price."""
    try:
        table_name = f"processed_{category.lower()}"
        df = fetch_table(table_name)
        
        if spec_name not in df.columns:
            return {"error": f"Spec '{spec_name}' not found in category '{category}'"}
        
        # Filter out non-numeric values if any
        numeric_df = df[['price', spec_name]].apply(pd.to_numeric, errors='coerce').dropna()
        
        if numeric_df.empty:
            return {"error": f"No numeric data found for spec '{spec_name}'"}
        
        correlation = numeric_df['price'].corr(numeric_df[spec_name])
        
        return {
            "category": category,
            "spec_name": spec_name,
            "correlation": float(correlation),
            "sample_size": len(numeric_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending/{category}")
def get_trending_products(category: str, limit: int = 10, sort_by: str = "review_count"):
    """Get trending products based on review count or rating."""
    valid_sort_fields = ["review_count", "rating"]
    
    if sort_by not in valid_sort_fields:
        raise HTTPException(status_code=400, detail=f"sort_by must be one of {valid_sort_fields}")
    
    try:
        query = f"""
        SELECT asin, title, price, rating, review_count, brand
        FROM products
        WHERE category = '{category}' AND {sort_by} IS NOT NULL
        ORDER BY {sort_by} DESC
        LIMIT {limit}
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": f"No products found for category: {category}"}
        
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discount/analysis/{category}")
def get_discount_analysis(category: str):
    """Analyze discounts for a specific category."""
    try:
        query = f"""
        SELECT 
            COUNT(*) as total_products,
            COUNT(CASE WHEN discount > 0 THEN 1 END) as discounted_products,
            AVG(CASE WHEN discount > 0 THEN discount END) as avg_discount,
            MAX(discount) as max_discount
        FROM products
        WHERE category = '{category}'
        """
        df = fetch_query(query)
        
        if df.empty:
            return {"error": f"No data found for category: {category}"}
        
        result = df.iloc[0].to_dict()
        
        # Calculate percentage of products on discount
        if result['total_products'] > 0:
            result['discount_percentage'] = round((result['discounted_products'] / result['total_products']) * 100, 2)
        else:
            result['discount_percentage'] = 0
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
