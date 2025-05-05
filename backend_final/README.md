Here is a detailed list of all the endpoints in the `backend_final` application, including their requirements, request formats, and response formats:

### 1. Analytics Endpoints

**Endpoint:** `/api/analytics/price-vs-rating`
- **Description:** Get price vs. rating data for scatter plot.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [
    {
      "price": float,
      "rating": float
    }
  ]
  ```

**Endpoint:** `/api/analytics/top-brands`
- **Description:** Get top brands by product count and average rating.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [
    {
      "brand": string,
      "count": int,
      "avg_rating": float
    }
  ]
  ```

**Endpoint:** `/api/analytics/discount-vs-rating`
- **Description:** Get discount vs. rating data.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [
    {
      "discount": float,
      "rating": float
    }
  ]
  ```

**Endpoint:** `/api/analytics/review-distribution`
- **Description:** Get review count distribution.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [int]
  ```

**Endpoint:** `/api/analytics/category-comparison`
- **Description:** Get category-wise comparison data (box plot values).
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [
    {
      "category": string,
      "avg_price": float,
      "avg_rating": float,
      "avg_discount": float
    }
  ]
  ```

**Endpoint:** `/api/analytics/product-addition-trend`
- **Description:** Get new product additions over time by category.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  {
    "date_str": {
      "category": int
    }
  }
  ```

**Endpoint:** `/api/analytics/price-trend`
- **Description:** Get price change over time (trend lines).
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  [
    {
      "date": string,
      "avg_price": float
    }
  ]
  ```

### 2. Brand Intelligence Endpoints

**Endpoint:** `/api/brand/avg-metrics`
- **Description:** Get brand-wise average price, rating, discount.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category` (optional): string
    - `limit` (optional): int (default: 10)
- **Response Format:**
  ```json
  [
    {
      "brand": string,
      "avg_price": float,
      "avg_rating": float,
      "avg_discount": float,
      "product_count": int
    }
  ]
  ```

**Endpoint:** `/api/brand/market-share`
- **Description:** Get market share by brand (product count within a category).
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
- **Response Format:**
  ```json
  [
    {
      "brand": string,
      "count": int,
      "percentage": float
    }
  ]
  ```

**Endpoint:** `/api/brand/brand-averages`
- **Description:** Get brand-wise average price, rating, discount with min_products filter.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
    - `min_products` (optional): int (default: 5)
- **Response Format:**
  ```json
  [
    {
      "brand": string,
      "product_count": int,
      "avg_price": float,
      "avg_rating": float,
      "avg_discount": float
    }
  ]
  ```

**Endpoint:** `/api/brand/brand-market-share`
- **Description:** Get market share by brand (based on product count).
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
- **Response Format:**
  ```json
  [
    {
      "brand": string,
      "count": int,
      "percentage": float
    }
  ]
  ```

### 3. Product Comparison Endpoints

**Endpoint:** `/api/compare/specs`
- **Description:** Compare specifications of multiple products side by side.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `product_ids`: List[int]
- **Response Format:**
  ```json
  {
    "products": [
      {
        "product_id": int,
        "title": string,
        "specs": {
          "spec_name": "spec_value"
        }
      }
    ]
  }
  ```

**Endpoint:** `/api/compare/spec-frequency`
- **Description:** Count frequency of a particular spec value across all products in a category.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `spec_name`: string
    - `category`: string
- **Response Format:**
  ```json
  [
    {
      "spec_value": string,
      "count": int
    }
  ]
  ```

### 4. Image-Driven Dashboards Endpoints

**Endpoint:** `/api/image-gallery/top-rated`
- **Description:** Get top-rated products with images.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
    - `limit` (optional): int (default: 10)
- **Response Format:**
  ```json
  [
    {
      "title": string,
      "rating": float,
      "price": float,
      "image_url": string
    }
  ]
  ```

**Endpoint:** `/api/image-gallery/best-discounts`
- **Description:** Get products with the highest discount to rating ratio.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
    - `limit` (optional): int (default: 10)
- **Response Format:**
  ```json
  [
    {
      "title": string,
      "discount": float,
      "rating": float,
      "price": float,
      "image_url": string
    }
  ]
  ```

### 5. Machine Learning Endpoints

**Endpoint:** `/ml/supervised-learning`
- **Description:** Run supervised learning algorithms (regression and classification).
- **Method:** POST
- **Requirements:**
  - Request Body:
    ```json
    {
      "features": [
        {
          "product_id": int,
          "price": float,
          "rating": float,
          "review_count": int,
          "category": string,
          "brand": string,
          "specs": {
            "spec_name": "spec_value"
          }
        }
      ],
      "target_column": string,
      "algorithm": string,
      "params": {
        "param_name": value
      },
      "test_size": float
    }
    ```
- **Response Format:**
  ```json
  {
    "success": bool,
    "metrics": {
      "metric_name": value
    },
    "detailed_report": object,
    "visualizations": {
      "visualization_name": base64_string
    },
    "feature_importance": {
      "feature_name": importance_value
    }
  }
  ```

**Endpoint:** `/ml/clustering`
- **Description:** Run clustering algorithms.
- **Method:** POST
- **Requirements:**
  - Request Body:
    ```json
    {
      "algorithm": string,
      "n_clusters": int,
      "params": {
        "param_name": value
      },
      "features": [string]
    }
    ```
- **Response Format:**
  ```json
  {
    "success": bool,
    "algorithm": string,
    "cluster_assignments": [int],
    "num_clusters": int,
    "product_ids": [int],
    "metrics": {
      "metric_name": value
    },
    "cluster_statistics": {
      "cluster_id": {
        "feature_name": {
          "mean": float,
          "median": float,
          "min": float,
          "max": float,
          "count": int
        }
      }
    },
    "visualizations": {
      "visualization_name": base64_string
    }
  }
  ```

**Endpoint:** `/ml/association-rules`
- **Description:** Run association rule mining.
- **Method:** POST
- **Requirements:**
  - Request Body:
    ```json
    {
      "min_support": float,
      "min_confidence": float,
      "min_lift": float,
      "metric": string,
      "category": string
    }
    ```
- **Response Format:**
  ```json
  {
    "success": bool,
    "num_rules": int,
    "rules": [
      {
        "antecedents": [string],
        "consequents": [string],
        "support": float,
        "confidence": float,
        "lift": float
      }
    ],
    "visualizations": {
      "visualization_name": base64_string
    }
  }
  ```

**Endpoint:** `/ml/available-algorithms`
- **Description:** Get the list of available algorithms with their parameters and descriptions.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  {
    "success": bool,
    "algorithms": {
      "supervised": {
        "regression": [
          {
            "name": string,
            "display_name": string,
            "description": string,
            "parameters": [
              {
                "name": string,
                "type": string,
                "default": value,
                "description": string
              }
            ]
          }
        ],
        "classification": [
          {
            "name": string,
            "display_name": string,
            "description": string,
            "parameters": [
              {
                "name": string,
                "type": string,
                "default": value,
                "description": string
              }
            ]
          }
        ]
      },
      "unsupervised": {
        "clustering": [
          {
            "name": string,
            "display_name": string,
            "description": string,
            "parameters": [
              {
                "name": string,
                "type": string,
                "default": value,
                "description": string
              }
            ]
          }
        ],
        "association_rules": [
          {
            "name": string,
            "display_name": string,
            "description": string,
            "parameters": [
              {
                "name": string,
                "type": string,
                "default": value,
                "description": string
              }
            ]
          }
        ]
      }
    }
  }
  ```

**Endpoint:** `/ml/model-compatibility`
- **Description:** Check if an algorithm is compatible with the current dataset.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `algorithm`: string
    - `target_column` (optional): string
- **Response Format:**
  ```json
  {
    "success": bool,
    "algorithm": string,
    "target_column": string,
    "compatible": bool,
    "reasons": [string],
    "dataset_info": {
      "num_samples": int,
      "num_features": int,
      "available_columns": [string]
    }
  }
  ```

**Endpoint:** `/ml/data-stats`
- **Description:** Get statistics about the available data for ML analysis.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  {
    "success": bool,
    "statistics": {
      "num_products": int,
      "num_categories": int,
      "num_brands": int,
      "price_range": [float, float],
      "rating_range": [float, float],
      "review_count_range": [int, int]
    },
    "specifications": {
      "total_specs": int,
      "top_specs": [
        {
          "name": string,
          "count": int
        }
      ]
    },
    "visualizations": {
      "visualization_name": base64_string
    }
  }
  ```

**Endpoint:** `/ml/feature-importance`
- **Description:** Calculate feature importance using Random Forest.
- **Method:** POST
- **Requirements:**
  - Query Parameters:
    - `target_column`: string
- **Response Format:**
  ```json
  {
    "success": bool,
    "target_column": string,
    "model_type": string,
    "feature_importance": [
      {
        "feature": string,
        "importance": float
      }
    ],
    "visualization": base64_string
  }
  ```

### 6. Product Endpoints

**Endpoint:** `/api/products/`
- **Description:** Get all products.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `skip` (optional): int (default: 0)
    - `limit` (optional): int (default: 100)
- **Response Format:**
  ```json
  [
    {
      "id": int,
      "title": string,
      "price": float,
      "rating": float,
      "category": string,
      "discount": float,
      "review_count": int
    }
  ]
  ```

**Endpoint:** `/api/products/by-category`
- **Description:** Get products by category.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
- **Response Format:**
  ```json
  [
    {
      "id": int,
      "title": string,
      "price": float,
      "rating": float,
      "category": string,
      "discount": float,
      "review_count": int
    }
  ]
  ```

**Endpoint:** `/api/products/top-rated`
- **Description:** Get top-rated products.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `limit` (optional): int (default: 10)
- **Response Format:**
  ```json
  [
    {
      "id": int,
      "title": string,
      "price": float,
      "rating": float,
      "category": string,
      "discount": float,
      "review_count": int
    }
  ]
  ```

**Endpoint:** `/api/products/search`
- **Description:** Search products by title.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `q`: string (min length: 2)
- **Response Format:**
  ```json
  [
    {
      "id": int,
      "title": string,
      "price": float,
      "rating": float,
      "category": string,
      "discount": float,
      "review_count": int
    }
  ]
  ```

### 7. Specification Insights Endpoints

**Endpoint:** `/api/specs/top-specs/{category}`
- **Description:** Get top N most common specs in a category.
- **Method:** GET
- **Requirements:**
  - Path Parameters:
    - `category`: string
  - Query Parameters:
    - `limit` (optional): int (default: 5)
- **Response Format:**
  ```json
  {
    "spec_name": [
      {
        "value": string,
        "count": int
      }
    ]
  }
  ```

**Endpoint:** `/api/specs/feature-frequency/{spec_keyword}`
- **Description:** Get feature presence frequency.
- **Method:** GET
- **Requirements:**
  - Path Parameters:
    - `spec_keyword`: string
- **Response Format:**
  ```json
  [
    {
      "category": string,
      "count": int
    }
  ]
  ```

**Endpoint:** `/api/specs/spec-vs-rating`
- **Description:** Get spec vs. rating correlation.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `spec_name`: string
- **Response Format:**
  ```json
  [
    {
      "spec_value": float,
      "rating": float
    }
  ]
  ```

**Endpoint:** `/api/specs/product-specs/{product_id}`
- **Description:** Get raw spec search for a given product.
- **Method:** GET
- **Requirements:**
  - Path Parameters:
    - `product_id`: int
- **Response Format:**
  ```json
  [
    {
      "name": string,
      "value": string
    }
  ]
  ```

### 8. Specification Endpoints

**Endpoint:** `/api/spec-insight/top-specs`
- **Description:** Get top N most common spec values for a given spec name and category.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category`: string
    - `spec_name`: string
    - `limit` (optional): int (default: 5)
- **Response Format:**
  ```json
  [
    {
      "spec_value": string,
      "count": int
    }
  ]
  ```

### 9. Trend Analysis Endpoints

**Endpoint:** `/api/trend/product-addition`
- **Description:** Get new product addition over time (daily/monthly).
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category` (optional): string
    - `group_by` (optional): string (default: "month")
- **Response Format:**
  ```json
  [
    {
      "time_unit": string,
      "count": int
    }
  ]
  ```

**Endpoint:** `/api/trend/avg-price`
- **Description:** Get average price trend over time.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category` (optional): string
    - `group_by` (optional): string (default: "month")
- **Response Format:**
  ```json
  [
    {
      "time_unit": string,
      "avg_price": float
    }
  ]
  ```

**Endpoint:** `/api/trend/rating-review`
- **Description:** Get rating and review count trend over time.
- **Method:** GET
- **Requirements:**
  - Query Parameters:
    - `category` (optional): string
    - `group_by` (optional): string (default: "month")
- **Response Format:**
  ```json
  [
    {
      "time_unit": string,
      "avg_rating": float,
      "total_reviews": int
    }
  ]
  ```

### 10. Root Endpoint

**Endpoint:** `/`
- **Description:** Welcome message.
- **Method:** GET
- **Requirements:** None
- **Response Format:**
  ```json
  {
    "message": "Welcome to the E-commerce Analytics API"
  }
  ```

This detailed list covers all the endpoints defined in the `backend_final` application, along with their requirements and response formats.