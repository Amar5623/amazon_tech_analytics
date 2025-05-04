# /ecom_api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import products, specs, analytics, models as ml_models

app = FastAPI(title="Amazon Tech Analytics API")

# Enable CORS (for dashboard access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(products.router, prefix="/api/products")
app.include_router(specs.router, prefix="/api/specs")
app.include_router(analytics.router, prefix="/api/analytics")
app.include_router(ml_models.router, prefix="/api/models")

@app.get("/")
def root():
    return {"message": "Welcome to Amazon Analytics API"}
