"""
Amazon Tech Gadget Data Preprocessing
-------------------------------------
This script cleans and structures Amazon product and specification data
without making assumptions or imputing values.
"""

import pandas as pd
import numpy as np
import re
import json
import os
from typing import Dict, List, Any

class AmazonDataPreprocessor:
    def __init__(self, products_df, specs_df):
        """
        Initialize with raw dataframes
        
        Args:
            products_df: DataFrame containing product information
            specs_df: DataFrame containing product specifications
        """
        self.products_df = products_df.copy()
        self.specs_df = specs_df.copy()
        self.processed_products = None
        self.processed_specs = None
        self.specs_pivoted = None
        self.common_specs = None
        
    def clean_products(self) -> pd.DataFrame:
        """Clean the products dataframe without altering core data"""
        df = self.products_df.copy()
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if not df[col].isna().all() else df[col]
        
        # Clean price: keep as numeric without currency symbols
        if 'price' in df.columns:
            # Handle potential string format with commas
            if df['price'].dtype == 'object':
                df['price'] = df['price'].astype(str).str.replace(',', '').str.extract(r'(\d+\.?\d*)').astype(float)
            # Already numeric, no action needed
        
        # Clean discount: extract numeric value from percentage string
        if 'discount' in df.columns:
            df['discount_value'] = df['discount'].str.extract(r'(\d+)').astype(float)
            # Keep original discount format for display purposes
        
        # Make sure rating is numeric
        if 'rating' in df.columns and df['rating'].dtype == 'object':
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Make sure review_count is numeric
        if 'review_count' in df.columns and df['review_count'].dtype == 'object':
            df['review_count'] = pd.to_numeric(df['review_count'].str.replace(',', ''), errors='coerce')
        
        # Extract brand from brand field (remove any hidden characters)
        if 'brand' in df.columns:
            df['brand_clean'] = df['brand'].str.replace(r'[^\x00-\x7F]+', '').str.strip()
        
        # Create simplified title field (but keep original)
        if 'title' in df.columns:
            # Keep original title but create a simplified version for analysis
            df['title_simplified'] = df['title'].str.replace(r'\([^)]*\)', ' ', regex=True)  # Remove parenthetical text
            df['title_simplified'] = df['title_simplified'].str.replace(r'[^\w\s]', ' ', regex=True)  # Remove special chars
            df['title_simplified'] = df['title_simplified'].str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
            df['title_simplified'] = df['title_simplified'].str.strip()
        
        # Extract ASIN from URL if missing in asin field
        if 'url' in df.columns and 'asin' in df.columns:
            missing_asin = df['asin'].isna()
            if any(missing_asin):
                # Extract ASIN from URL for rows with missing ASIN
                df.loc[missing_asin, 'asin'] = df.loc[missing_asin, 'url'].str.extract(r'/dp/([A-Z0-9]{10})')
        
        self.processed_products = df
        return df
    
    def clean_specifications(self) -> pd.DataFrame:
        """Clean the specifications dataframe without altering core data"""
        df = self.specs_df.copy()
        
        # Strip whitespace and special characters from spec names and values
        for col in ['spec_name', 'spec_value']:
            if col in df.columns:
                # Remove non-printable characters and leading/trailing whitespace
                df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '').str.strip()
        
        # Normalize spec names: convert to lowercase, replace spaces with underscores
        if 'spec_name' in df.columns:
            df['spec_name_normalized'] = df['spec_name'].str.lower().str.replace(r'\s+', '_', regex=True)
            df['spec_name_normalized'] = df['spec_name_normalized'].str.replace(r'[^\w_]', '', regex=True)
        
        # Clean HTML from spec values
        if 'spec_value' in df.columns:
            df['spec_value_clean'] = df['spec_value'].str.replace(r'<[^>]+>', ' ', regex=True)
            df['spec_value_clean'] = df['spec_value_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Extract numeric values where possible (for analytical purposes)
        if 'spec_value' in df.columns:
            # Create a column that extracts just numeric values for potential analysis
            df['numeric_value'] = df['spec_value'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        self.processed_specs = df
        return df
    
    def extract_key_specifications(self) -> pd.DataFrame:
        """
        Extract key specifications useful for dashboards
        Focus on common, analytically useful specifications
        """
        if self.processed_specs is None:
            self.clean_specifications()
        
        df = self.processed_specs
        
        # Count occurrence of each spec to find common ones
        spec_counts = df['spec_name_normalized'].value_counts()
        
        # Consider specs that appear in more than 30% of products as "common"
        product_count = self.processed_products['id'].nunique()
        common_spec_threshold = max(3, int(product_count * 0.3))  # At least 3 products
        common_specs = spec_counts[spec_counts >= common_spec_threshold].index.tolist()
        
        # Store common specs for reference
        self.common_specs = common_specs
        
        # Filter for important analytical specs (even if not common)
        important_spec_patterns = [
            'processor', 'cpu', 'core', 'speed', 'ghz', 
            'memory', 'ram', 'storage', 'ssd', 'hdd', 'gb', 'tb',
            'screen', 'display', 'resolution', 'inch', 
            'graphics', 'gpu', 'video',
            'battery', 'weight', 'dimension'
        ]
        
        # Find specs matching important patterns
        important_specs = []
        for pattern in important_spec_patterns:
            matches = [s for s in df['spec_name_normalized'].unique() if pattern in s]
            important_specs.extend(matches)
        
        # Combine common and important specs
        key_specs = list(set(common_specs + important_specs))
        
        # Create a filtered dataframe with key specs
        key_specs_df = df[df['spec_name_normalized'].isin(key_specs)]
        
        return key_specs_df
    
    def pivot_specifications(self) -> pd.DataFrame:
        """
        Create a pivoted view of specifications with products as rows
        and specification names as columns
        """
        if self.processed_specs is None:
            self.clean_specifications()
            
        # Extract key specs to avoid overwhelming column count
        key_specs = self.extract_key_specifications()
        
        # Create pivot table
        pivot_df = key_specs.pivot_table(
            index='product_id',
            columns='spec_name_normalized',
            values='spec_value_clean',
            aggfunc='first'  # In case of duplicates, take first value
        ).reset_index()
        
        # Rename columns to make them more readable
        pivot_df.columns = [col.replace('_', ' ').title() if col != 'product_id' else col for col in pivot_df.columns]
        
        self.specs_pivoted = pivot_df
        return pivot_df
    
    def create_analytical_dataset(self) -> pd.DataFrame:
        """
        Join processed products with pivoted specifications
        to create a unified dataset for analysis
        """
        if self.processed_products is None:
            self.clean_products()
            
        if self.specs_pivoted is None:
            self.pivot_specifications()
            
        # Merge products with pivoted specifications
        analytical_df = pd.merge(
            self.processed_products,
            self.specs_pivoted,
            left_on='id',
            right_on='product_id',
            how='left'
        )
        
        return analytical_df
    
    def extract_structured_specs(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract structured specifications as nested dictionaries
        for each product, useful for JSON API responses
        """
        if self.processed_specs is None:
            self.clean_specifications()
            
        structured_specs = {}
        
        # Group by product_id
        grouped = self.processed_specs.groupby('product_id')
        
        for product_id, group in grouped:
            # Create dict of spec_name: spec_value for this product
            product_specs = dict(zip(
                group['spec_name_normalized'], 
                group['spec_value_clean']
            ))
            
            # For certain specs, try to further structure the data
            structured_product_specs = self._structure_product_specs(product_specs)
            structured_specs[product_id] = structured_product_specs
            
        return structured_specs
    
    def _structure_product_specs(self, specs: Dict[str, str]) -> Dict[str, Any]:
        """Helper function to organize specs into meaningful categories"""
        structured = {
            'general': {},
            'processor': {},
            'memory': {},
            'display': {},
            'storage': {},
            'graphics': {},
            'physical': {},
            'connectivity': {},
            'other': {}
        }
        
        # Map specs to categories based on keywords
        for spec_name, value in specs.items():
            if any(x in spec_name for x in ['processor', 'cpu', 'core']):
                structured['processor'][spec_name] = value
            elif any(x in spec_name for x in ['memory', 'ram']):
                structured['memory'][spec_name] = value
            elif any(x in spec_name for x in ['display', 'screen', 'resolution']):
                structured['display'][spec_name] = value
            elif any(x in spec_name for x in ['storage', 'ssd', 'hdd', 'drive']):
                structured['storage'][spec_name] = value
            elif any(x in spec_name for x in ['graphics', 'gpu', 'video']):
                structured['graphics'][spec_name] = value
            elif any(x in spec_name for x in ['weight', 'dimension', 'size']):
                structured['physical'][spec_name] = value
            elif any(x in spec_name for x in ['wifi', 'bluetooth', 'usb', 'port']):
                structured['connectivity'][spec_name] = value
            elif any(x in spec_name for x in ['brand', 'manufacturer', 'model', 'series']):
                structured['general'][spec_name] = value
            else:
                structured['other'][spec_name] = value
        
        # Remove empty categories
        return {k: v for k, v in structured.items() if v}
    
    def save_processed_data(self, output_dir: str = 'data/processed'):
        """Save all processed dataframes to the specified directory"""
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process data if not already done
        if self.processed_products is None:
            self.clean_products()
        
        if self.processed_specs is None:
            self.clean_specifications()
            
        if self.specs_pivoted is None:
            self.pivot_specifications()
            
        # Save files
        self.processed_products.to_csv(f'{output_dir}/processed_products.csv', index=False)
        self.processed_specs.to_csv(f'{output_dir}/processed_specs.csv', index=False)
        self.specs_pivoted.to_csv(f'{output_dir}/specs_pivoted.csv', index=False)
        
        # Save analytical dataset
        analytical_df = self.create_analytical_dataset()
        analytical_df.to_csv(f'{output_dir}/analytical_dataset.csv', index=False)
        
        # Save structured specs as JSON
        structured_specs = self.extract_structured_specs()
        with open(f'{output_dir}/structured_specs.json', 'w') as f:
            json.dump(structured_specs, f)
            
        print(f"All processed data saved to {output_dir}/")
        
    def extract_category_insights(self) -> pd.DataFrame:
        """
        Extract category-level insights like price ranges,
        average ratings, common specifications, etc.
        """
        if self.processed_products is None:
            self.clean_products()
            
        df = self.processed_products
        
        # Group by category
        if 'category' in df.columns:
            category_insights = df.groupby('category').agg({
                'id': 'count',  # Count of products
                'price': ['min', 'max', 'mean', 'median'],  # Price statistics
                'rating': ['mean', 'median', 'count'],  # Rating statistics
                'review_count': ['sum', 'mean']  # Review statistics
            })
            
            # Flatten MultiIndex
            category_insights.columns = ['_'.join(col).strip() for col in category_insights.columns.values]
            category_insights = category_insights.reset_index()
            
            return category_insights
        
        return pd.DataFrame()  # Return empty DataFrame if category column doesn't exist
    
    def extract_brand_insights(self) -> pd.DataFrame:
        """
        Extract brand-level insights like product counts,
        average prices, ratings, etc.
        """
        if self.processed_products is None:
            self.clean_products()
            
        df = self.processed_products
        brand_col = 'brand_clean' if 'brand_clean' in df.columns else 'brand'
        
        # Group by brand
        if brand_col in df.columns:
            brand_insights = df.groupby(brand_col).agg({
                'id': 'count',  # Count of products
                'price': ['min', 'max', 'mean', 'median'],  # Price statistics
                'rating': ['mean', 'median', 'count'],  # Rating statistics
                'review_count': ['sum', 'mean'],  # Review statistics
                'category': lambda x: x.value_counts().index[0] if not x.empty else None  # Most common category
            })
            
            # Flatten MultiIndex
            brand_insights.columns = ['_'.join(col).strip() for col in brand_insights.columns.values]
            brand_insights = brand_insights.reset_index()
            
            return brand_insights
        
        return pd.DataFrame()  # Return empty DataFrame if brand column doesn't exist
    
    def prepare_dashboard_datasets(self, output_dir: str = 'data/dashboard'):
        """
        Prepare specific datasets optimized for each dashboard view
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Price Comparison Dataset
        if self.processed_products is not None:
            price_df = self.processed_products[['id', 'title_simplified', 'brand_clean', 'category', 'price', 'discount_value', 'rating']]
            price_df.to_csv(f'{output_dir}/price_comparison.csv', index=False)
        
        # 2. Category Analysis Dataset
        category_insights = self.extract_category_insights()
        if not category_insights.empty:
            category_insights.to_csv(f'{output_dir}/category_analysis.csv', index=False)
        
        # 3. Brand Analysis Dataset
        brand_insights = self.extract_brand_insights()
        if not brand_insights.empty:
            brand_insights.to_csv(f'{output_dir}/brand_analysis.csv', index=False)
        
        # 4. Rating Distribution Dataset
        if self.processed_products is not None:
            rating_df = self.processed_products[['id', 'title_simplified', 'brand_clean', 'category', 'rating', 'review_count']]
            rating_df.to_csv(f'{output_dir}/rating_distribution.csv', index=False)
        
        # 5. Specification Comparison Dataset
        if self.specs_pivoted is not None:
            # Join with essential product info
            if self.processed_products is not None:
                spec_comparison = pd.merge(
                    self.specs_pivoted,
                    self.processed_products[['id', 'title_simplified', 'brand_clean', 'category', 'price']],
                    left_on='product_id',
                    right_on='id',
                    how='left'
                )
                spec_comparison.to_csv(f'{output_dir}/spec_comparison.csv', index=False)
        
        print(f"Dashboard-specific datasets saved to {output_dir}/")


# Example usage (to be replaced with actual loading of your data)
def load_data_from_csv():
    """Load sample data from CSV files"""
    try:
        products_df = pd.read_csv('scraper/data/raw/products.csv')
        specs_df = pd.read_csv('scraper/data/raw/specifications.csv')
        return products_df, specs_df
    except FileNotFoundError:
        print("Data files not found. Please place your CSV files in the data/raw directory.")
        return None, None

def main():
    # Load data
    products_df, specs_df = load_data_from_csv()
    
    if products_df is None or specs_df is None:
        return
    
    # Initialize preprocessor
    preprocessor = AmazonDataPreprocessor(products_df, specs_df)
    
    # Clean and process data
    preprocessor.clean_products()
    preprocessor.clean_specifications()
    
    # Create pivoted view of specifications
    preprocessor.pivot_specifications()
    
    # Save all processed data
    preprocessor.save_processed_data()
    
    # Prepare dashboard-specific datasets
    preprocessor.prepare_dashboard_datasets()
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()