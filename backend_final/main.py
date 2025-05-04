# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (
    products, analytics, ml, specs,
    brand_router, comparison, image_gallery,
    spec_insight, trend
)

# Initialize the FastAPI app
app = FastAPI()

# CORS Middleware setup for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development, adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for different functionalities
app.include_router(products.router, prefix="/products", tags=["products"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(ml.router, prefix="/ml", tags=["ml"])
app.include_router(specs.router, prefix="/specs", tags=["specifications"])
app.include_router(comparison.router, prefix="/compare", tags=["comparison"])
app.include_router(brand_router.router, prefix="/brand", tags=["brand"])
app.include_router(image_gallery.router, prefix="/image-gallery", tags=["image gallery"])
app.include_router(spec_insight.router, prefix="/spec-insight", tags=["specification insight"])
app.include_router(trend.router, prefix="/trend", tags=["trend"])

@app.get("/")
async def root():
    return {"message": "Welcome to the E-commerce Analytics API"}
