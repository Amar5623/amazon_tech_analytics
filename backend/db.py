# /ecom_api/db.py
from sqlalchemy import create_engine
import pandas as pd

DB_URL = "mysql+pymysql://root:root@localhost:3306/amazon_scraper_cleaned"
engine = create_engine(DB_URL)

def fetch_table(table_name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

def fetch_query(query: str) -> pd.DataFrame:
    return pd.read_sql(query, con=engine)