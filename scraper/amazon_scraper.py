import requests
from bs4 import BeautifulSoup
from lxml import etree, html
import re
import json
import time
import random
import logging
import mysql.connector
from mysql.connector import pooling
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("amazon_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and operation manager"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database connection pool
        
        Args:
            config: Database connection configuration
        """
        self.config = config
        self.connection_pool = None
        self._create_tables_if_not_exist()
    
    def _get_connection(self):
        """Get a connection from the pool or create a new one"""
        if not self.connection_pool:
            try:
                self.connection_pool = pooling.MySQLConnectionPool(
                    pool_name="amazon_scraper_pool",
                    pool_size=5,
                    **self.config
                )
                logger.info("Created database connection pool")
            except mysql.connector.Error as err:
                logger.error(f"Error creating connection pool: {err}")
                raise
        
        try:
            return self.connection_pool.get_connection()
        except mysql.connector.Error as err:
            logger.error(f"Error getting connection from pool: {err}")
            raise
    
    def _create_tables_if_not_exist(self):
        """Create database tables if they don't exist"""
        conn = mysql.connector.connect(**self.config)
        try:
            cursor = conn.cursor()
            
            # Create products table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                asin VARCHAR(20) UNIQUE,
                url VARCHAR(1000) NOT NULL,
                title VARCHAR(500),
                price DECIMAL(10, 2),
                discount VARCHAR(50),
                rating DECIMAL(2, 1),
                review_count INT,
                brand VARCHAR(100),
                category VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            # Create specifications table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS specifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT,
                spec_name VARCHAR(255) NOT NULL,
                spec_value TEXT,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
                UNIQUE KEY unique_spec (product_id, spec_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            # Create images table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT,
                image_url VARCHAR(1000) NOT NULL,
                position INT,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            # Create scraped_data table for raw JSON backup
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                asin VARCHAR(20) UNIQUE,
                raw_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """)
            
            conn.commit()
            logger.info("Database tables created or already exist")
        except mysql.connector.Error as err:
            logger.error(f"Error creating tables: {err}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def insert_product(self, product_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert product data into database
        
        Args:
            product_data: Dictionary containing product data
            
        Returns:
            Product ID if successful, None otherwise
        """
        conn = self._get_connection()
        product_id = None
        
        try:
            cursor = conn.cursor()
            
            # Extract ASIN from URL
            asin = None
            url = product_data.get('url', '')
            asin_match = re.search(r'/dp/([A-Z0-9]{10})/', url)
            if asin_match:
                asin = asin_match.group(1)
            
            # First check if product exists
            if asin:
                cursor.execute("SELECT id FROM products WHERE asin = %s", (asin,))
                result = cursor.fetchone()
                
                if result:
                    product_id = result[0]
                    logger.info(f"Product already exists with ID {product_id}, updating")
                    
                    # Update product data
                    cursor.execute("""
                    UPDATE products SET 
                        title = %s,
                        price = %s,
                        discount = %s,
                        rating = %s,
                        review_count = %s,
                        brand = %s,
                        category = %s
                    WHERE id = %s
                    """, (
                        product_data.get('title'),
                        self._parse_price(product_data.get('price')),
                        product_data.get('discount'),
                        self._parse_rating(product_data.get('rating')),
                        self._parse_review_count(product_data.get('review_count')),
                        product_data.get('brand'),
                        product_data.get('category'),
                        product_id
                    ))
                    
                    # Delete existing specifications to replace them
                    cursor.execute("DELETE FROM specifications WHERE product_id = %s", (product_id,))
                    
                    # Delete existing images to replace them
                    cursor.execute("DELETE FROM images WHERE product_id = %s", (product_id,))
            
            # Insert new product if it doesn't exist
            if not product_id:
                cursor.execute("""
                INSERT INTO products (
                    asin, url, title, price, discount, rating, review_count, brand, category
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    asin,
                    product_data.get('url'),
                    product_data.get('title'),
                    self._parse_price(product_data.get('price')),
                    product_data.get('discount'),  # Add this line
                    self._parse_rating(product_data.get('rating')),
                    self._parse_review_count(product_data.get('review_count')),
                    product_data.get('brand'),
                    product_data.get('category')
                ))
                product_id = cursor.lastrowid
                logger.info(f"Inserted new product with ID {product_id}")
            
            # Insert specifications
            specs = product_data.get('specs', [])
            if specs and isinstance(specs, list) and len(specs) > 0:
                spec_dict = specs[0] if isinstance(specs[0], dict) else {}
                for spec_name, spec_value in spec_dict.items():
                    cursor.execute("""
                    INSERT INTO specifications (product_id, spec_name, spec_value)
                    VALUES (%s, %s, %s)
                    """, (product_id, spec_name[:255], str(spec_value)))
            
            # Insert images
            images = product_data.get('images', [])
            for position, image_url in enumerate(images):
                cursor.execute("""
                INSERT INTO images (product_id, image_url, position)
                VALUES (%s, %s, %s)
                """, (product_id, image_url[:1000], position))
            
            # Store raw JSON data for backup
            raw_json = json.dumps(product_data)
            cursor.execute("""
            INSERT INTO scraped_data (asin, raw_data)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE raw_data = %s, updated_at = CURRENT_TIMESTAMP
            """, (asin, raw_json, raw_json))
            
            conn.commit()
            return product_id
            
        except mysql.connector.Error as err:
            logger.error(f"Database error inserting product: {err}")
            conn.rollback()
            return None
        finally:
            cursor.close()
            conn.close()
    
    def _parse_price(self, price_str: Optional[str]) -> Optional[float]:
        """Parse price string to float"""
        if not price_str:
            return None
        
        # Remove currency symbols and commas
        price_str = re.sub(r'[^\d.]', '', price_str)
        
        try:
            return float(price_str) if price_str else None
        except ValueError:
            return None
    
    def _parse_rating(self, rating_str: Optional[str]) -> Optional[float]:
        """Parse rating string to float"""
        if not rating_str:
            return None
        
        # Extract the first number (e.g., "4.5 out of 5 stars" -> 4.5)
        match = re.search(r'(\d+(\.\d+)?)', rating_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def _parse_review_count(self, review_count_str: Optional[str]) -> Optional[int]:
        """Parse review count string to integer"""
        if not review_count_str:
            return None
        
        # Extract digits only
        digits = re.sub(r'[^\d]', '', review_count_str)
        
        try:
            return int(digits) if digits else None
        except ValueError:
            return None


class AmazonScraper:
    def __init__(self, base_url: str = "https://www.amazon.in", max_retries: int = 3, delay_range: tuple = (2, 5), db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Amazon scraper.
        
        Args:
            base_url: The base Amazon URL
            max_retries: Maximum number of retry attempts for failed requests
            delay_range: Tuple of (min_delay, max_delay) in seconds between requests
            db_config: Database configuration dictionary
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.min_delay, self.max_delay = delay_range
        self.session = requests.Session()
        self.headers = {
            "accept-language": "en-US,en;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": base_url
        }
        # User-Agent pool (you can add more)
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.139 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]

        # Proxy pool (example proxies — replace with real ones)
        self.proxies = []
        
        # Initialize database manager if config provided
        self.db_manager = DatabaseManager(db_config) if db_config else None
        
    def _random_delay(self) -> None:
        """Add a random delay between requests to avoid being blocked"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
        
    def _get_html(self, url: str) -> Optional[str]:
        """
        Fetch HTML content from a URL with retry mechanism, rotating headers and optionally using proxies.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content as string or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Rotate User-Agent
                current_headers = self.headers.copy()
                current_headers["User-Agent"] = random.choice(self.user_agents)

                # Use a random proxy only if available
                if self.proxies:
                    proxy = random.choice(self.proxies)
                    proxies = {"http": proxy, "https": proxy}
                    logger.info(f"Using proxy: {proxy}")
                else:
                    proxies = None

                # Make the request
                response = self.session.get(url, headers=current_headers, proxies=proxies, timeout=(10, 20))

                if response.status_code == 200:
                    return response.text
                elif response.status_code in [429, 503]:
                    logger.warning(f"Rate limited or blocked ({response.status_code}) on attempt {attempt+1} for {url}")
                else:
                    logger.error(f"HTTP error {response.status_code} on attempt {attempt+1} for {url}")

            except requests.RequestException as e:
                logger.error(f"Request error on attempt {attempt+1} for {url}: {str(e)}")

            # Wait before retrying
            self._random_delay()

        logger.error(f"All {self.max_retries} attempts failed for URL: {url}")
        return None
    
    def extract_product_links(self, search_html: str) -> List[str]:
        """
        Extract product links from a search results page using LXML XPath.
        
        Args:
            search_html: HTML content of the search results page
            
        Returns:
            List of product URLs
        """
        if not search_html:
            return []
            
        # Parse HTML into lxml element tree
        html_tree = html.fromstring(search_html)
        product_links = []
        
        # XPath expressions for product links
        xpath_expressions = [
            # Method 1: Data component search results
            "//div[contains(@data-component-type, 's-search-result')]//h2/a/@href",
            # Method 2: Fallback for standard layout
            "//div[contains(@class, 's-result-item')]//h2/a[contains(@class, 'a-link-normal')]/@href",
            # Method 3: Additional fallback
            "//div[contains(@class, 's-result-item')]//a[contains(@class, 'a-link-normal s-no-outline')]/@href"
        ]
        
        # Try each XPath expression
        for xpath in xpath_expressions:
            links = html_tree.xpath(xpath)
            if links:
                for href in links:
                    if '/dp/' in href:
                        product_url = urljoin(self.base_url, href)
                        product_links.append(product_url)
                
                if product_links:
                    break
        
        # Remove duplicates while preserving order
        unique_links = []
        seen = set()
        for link in product_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
                
        logger.info(f"Extracted {len(unique_links)} product links from search page")
        return unique_links
    
    def extract_product_data(self, product_html: str, product_url: str, category: str = "") -> Dict[str, Any]:
        if not product_html:
            logger.error(f"No HTML content to extract for {product_url}")
            return {}
        
        # Parse HTML
        soup = BeautifulSoup(product_html, 'lxml')
        tree = html.fromstring(product_html)
        
        # Initialize product data
        product_data = {
            "url": product_url,
            "title": None,
            "price": None,
            "discount": None,
            "rating": None,
            "review_count": None,
            "brand": None,
            "category": category,
            "images": [],
            "specs": []
        }
        
        # Extract ASIN from URL
        asin_match = re.search(r'/dp/([A-Z0-9]{10})/', product_url)
        if asin_match:
            product_data["asin"] = asin_match.group(1)
        
        # Extract title using XPath
        title_xpath_list = [
            "//*[@id='productTitle']/text()"
            "//span[@id='productTitle']/text()",
            "//h1[@id='title']//text()"
        ]
        
        for xpath in title_xpath_list:
            title_elements = tree.xpath(xpath)
            if title_elements:
                product_data["title"] = title_elements[0].strip()
                break
        
        # Extract price using XPath
        price_xpath_list = [
            "//*[@id='corePriceDisplay_desktop_feature_div']/div[1]/span[3]/span[2]/span[2]"
            "//span[contains(@class, 'a-price-whole')]/text()",
            "//span[@class='a-price']/span[@class='a-offscreen']/text()",
            "//span[@id='priceblock_ourprice']/text()"
        ]
        
        for xpath in price_xpath_list:
            price_elements = tree.xpath(xpath)
            if price_elements:
                product_data["price"] = price_elements[0].strip()
                break
        
        # Extract discount using XPath
        discount_xpath_list = [
            "//*[@id='corePriceDisplay_desktop_feature_div']/div[1]/span[2]"
            "//span[contains(@class, 'savingsPercentage')]/text()",
            "//td[contains(@class, 'priceBlockSavingsString')]/text()",
            "//span[contains(@class, 'a-color-price')]/text()"
        ]
        
        for xpath in discount_xpath_list:
            discount_elements = tree.xpath(xpath)
            if discount_elements:
                for discount_text in discount_elements:
                    discount_match = re.search(r'(\d+(?:\.\d+)?)%', discount_text)
                    if discount_match:
                        product_data["discount"] = discount_match.group(0).strip()
                        break
                if product_data["discount"]:
                    break
        
        # Alternative method using regular expressions directly on the HTML
        if not product_data["discount"]:
            discount_patterns = [
                r'Save:\s*(\d+(?:\.\d+)?)%',
                r'(\d+(?:\.\d+)?)% off',
                r'discount of (\d+(?:\.\d+)?)%'
            ]
            
            for pattern in discount_patterns:
                discount_match = re.search(pattern, product_html)
                if discount_match:
                    product_data["discount"] = f"{discount_match.group(1)}%"
                    break
        
        # Extract rating using XPath
        rating_xpath_list = [
            "//*[@id='acrPopover']/span[1]/a/span/text()"
            "//span[contains(@class, 'a-icon-alt')]/text()",
            "//i[contains(@class, 'a-icon-star')]/span/text()"
        ]
        
        for xpath in rating_xpath_list:
            rating_elements = tree.xpath(xpath)
            if rating_elements:
                product_data["rating"] = rating_elements[0].strip()
                break
        
        # Extract review count using XPath
        review_count_xpath_list = [
            "//*[@id='acrCustomerReviewText']/text()"
            "//span[@id='acrCustomerReviewText']/text()",
            "//a[contains(@class, 'a-link-normal') and contains(@href, '#customerReviews')]/span/text()"
        ]
        
        for xpath in review_count_xpath_list:
            review_elements = tree.xpath(xpath)
            if review_elements:
                product_data["review_count"] = review_elements[0].strip()
                break
        
        # Extract brand using XPath
        brand_xpath_list = [
            "//*[@id='poExpander']/div[1]/div/table/tbody/tr[1]/td[2]/span/text()"
            "//a[@id='bylineInfo']/text()",
            "//*[@id='productDetails_techSpec_section_1']/tbody/tr[1]/td",
            "//a[contains(@class, 'contributorNameID')]/text()",
            "//tr[th[contains(text(), 'Brand')]]/td/text()"
        ]
        
        for xpath in brand_xpath_list:
            brand_elements = tree.xpath(xpath)
            if brand_elements:
                brand_text = brand_elements[0].strip()
                brand_text = re.sub(r'^Brand:\s*', '', brand_text)
                product_data["brand"] = brand_text
                break
        
        # Extract images using XPath and regex approach (keep both methods)
        images = []
        
        # Method 1: Using regex to find image URLs in the page source
        image_matches = re.findall(r'"hiRes":"(.+?)"', product_html)
        if image_matches:
            images.extend(image_matches)
        
        # Method 2: Using XPath to find image URLs
        image_xpath_list = [
            "//div[@id='imgTagWrapperId']//img/@data-old-hires",
            "//div[@id='imageBlock']//img/@data-old-hires",
            "//div[@id='altImages']//img/@src"
        ]
        
        for xpath in image_xpath_list:
            image_elements = tree.xpath(xpath)
            if image_elements:
                for img_url in image_elements:
                    if img_url and 'images/I/' in img_url and not img_url.endswith('.gif'):
                        images.append(img_url)
        
        # Clean up image URLs and remove duplicates
        cleaned_images = []
        seen_images = set()
        for img_url in images:
            if img_url and img_url not in seen_images and not img_url.endswith('.gif'):
                seen_images.add(img_url)
                cleaned_images.append(img_url)
                
        product_data["images"] = cleaned_images
        
        # Extract specifications using XPath
        specs_dict = {}
        
        # Method 1: Product details table
        spec_table_xpath_list = [
            "//table[@id='productDetails_techSpec_section_1']//tr",
            "//table[@id='productDetails_detailBullets_sections1']//tr",
            "//table[contains(@class, 'prodDetTable')]//tr"
        ]
        
        for xpath in spec_table_xpath_list:
            spec_rows = tree.xpath(xpath)
            for row in spec_rows:
                header = row.xpath("./th")
                value = row.xpath("./td")
                if header and value:
                    key = header[0].text_content().strip()
                    val = value[0].text_content().strip()
                    if key and val:
                        specs_dict[key] = val
        
        # Method 2: Detail bullets
        bullet_xpath_list = [
            "//div[@id='detailBullets_feature_div']//li/span"
        ]
        
        for xpath in bullet_xpath_list:
            bullet_elements = tree.xpath(xpath)
            for bullet in bullet_elements:
                spans = bullet.xpath("./span")
                if len(spans) >= 2:
                    key = spans[0].text_content().replace("\u200e", "").replace(":", "").strip()
                    val = spans[1].text_content().strip()
                    if key and val:
                        specs_dict[key] = val
        
        # Method 3: Feature bullets
        feature_bullets = tree.xpath("//div[@id='feature-bullets']//li//span[@class='a-list-item']/text()")
        if feature_bullets:
            features = [bullet.strip() for bullet in feature_bullets if bullet.strip()]
            if features:
                specs_dict["Features"] = features
        
        product_data["specs"] = [specs_dict] if specs_dict else []
        
        # Log missing data
        for key, value in product_data.items():
            if value is None:
                logger.warning(f"Missing data for {key} in product {product_url}")
        
        return product_data
    
    def scrape_search_page(self, search_url: str, page_num: int = 1) -> Optional[str]:
        """
        Scrape a single search results page.
        
        Args:
            search_url: Base search URL
            page_num: Page number to scrape
            
        Returns:
            HTML content of the search page
        """
        if page_num > 1:
            # Add page parameter to URL
            if "?" in search_url:
                paginated_url = f"{search_url}&page={page_num}"
            else:
                paginated_url = f"{search_url}?page={page_num}"
        else:
            paginated_url = search_url
            
        logger.info(f"Scraping search page {page_num}: {paginated_url}")
        html_content = self._get_html(paginated_url)
        
        self._random_delay()
        return html_content
    
    def scrape_product_page(self, product_url: str, category: str = "") -> Dict[str, Any]:
        """
        Scrape a single product page.
        
        Args:
            product_url: URL of the product page
            category: Product category
            
        Returns:
            Dictionary containing product data
        """
        logger.info(f"Scraping product: {product_url}")
        html_content = self._get_html(product_url)
        
        if not html_content:
            logger.error(f"Failed to get HTML for product: {product_url}")
            return {}
            
        product_data = self.extract_product_data(html_content, product_url, category)
        
        # Save to database if database manager is initialized
        if self.db_manager and product_data:
            product_id = self.db_manager.insert_product(product_data)
            if product_id:
                logger.info(f"Saved product to database with ID: {product_id}")
        
        self._random_delay()
        return product_data
    
    def scrape_category(self, category_name: str, category_url: str, max_pages: int = 3, max_products: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape products from a category.
        
        Args:
            category_name: Name of the category
            category_url: URL of the category search page
            max_pages: Maximum number of search result pages to scrape
            max_products: Maximum number of products to scrape
            
        Returns:
            List of product data dictionaries
        """
        all_products = []
        product_urls = []
        
        # Step 1: Collect product URLs from search pages
        for page_num in range(1, max_pages + 1):
            search_html = self.scrape_search_page(category_url, page_num)
            
            if not search_html:
                logger.error(f"Failed to get search results for page {page_num}")
                continue
                
            page_product_urls = self.extract_product_links(search_html)
            product_urls.extend(page_product_urls)
            
            logger.info(f"Collected {len(page_product_urls)} product URLs from page {page_num}")
            
            # Check if we have enough product URLs
            if len(product_urls) >= max_products:
                product_urls = product_urls[:max_products]
                break
        
        # Step 2: Scrape individual product pages
        for i, product_url in enumerate(product_urls):
            logger.info(f"Processing product {i+1}/{len(product_urls)}")
            product_data = self.scrape_product_page(product_url, category_name)
            
            # Only add non-empty product data
            if product_data and product_data.get("title"):
                all_products.append(product_data)
                logger.info(f"Successfully scraped product: {product_data.get('title', 'Unknown')}")
            else:
                logger.warning(f"Failed to extract data for product URL: {product_url}")
            
            # Check if we've reached the maximum number of products
            if len(all_products) >= max_products:
                logger.info(f"Reached maximum number of products ({max_products})")
                break
        
        logger.info(f"Completed scraping {len(all_products)} products from category {category_name}")
        return all_products
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str) -> None:
        """
        Save scraped data to a JSON file.
        
        Args:
            data: List of product dictionaries
            filename: Name of the output file
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data successfully saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {str(e)}")


def main():
    """Main function to run the scraper"""
    # Database configuration
    db_config = {
        "host": "localhost",
        "user": "root",          # Replace with your MySQL username
        "password": "root",  # Replace with your MySQL password
        "database": "amazon_scraper_final"  # Database name
    }
    
    # Try to create the database if it doesn't exist
    try:
        temp_conn = mysql.connector.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"]
        )
        cursor = temp_conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        cursor.close()
        temp_conn.close()
        logger.info(f"Database {db_config['database']} created or already exists")
    except mysql.connector.Error as err:
        logger.error(f"Error creating database: {err}")
        return
    
    # Configuration
    categories = {
        "laptops": "https://www.amazon.in/s?k=laptops&crid=1N54BM2KAM741",
        "smartphones": "https://www.amazon.in/s?k=smartphones&crid=O3OUI1M5CM3H",
        "wireless_earbuds": "https://www.amazon.in/s?k=wireless+earbuds&crid=1XXEDHRYTQIVC",
        "headphones": "https://www.amazon.in/s?k=headphones&crid=156ZGTMASKEJ3",
    }
    
    max_pages_per_category = 1  # Number of search result pages to scrape per category
    max_products_per_category = 10  # Maximum number of products to scrape per category
    
    # Create output directory for JSON files
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Initialize the scraper
    scraper = AmazonScraper(
        base_url="https://www.amazon.in",
        max_retries=3,
        delay_range=(2, 5),  # Random delay between 3 and 7 seconds
        db_config=db_config
    )
    
    # Track overall statistics
    total_products = 0
    start_time = datetime.now()
    
    # Process each category
    for category_name, category_url in categories.items():
        try:
            logger.info(f"Starting to scrape category: {category_name}")
            
            # Scrape products from the category
            category_products = scraper.scrape_category(
                category_name=category_name,
                category_url=category_url,
                max_pages=max_pages_per_category,
                max_products=max_products_per_category
            )
            
            # Save category products to JSON
            if category_products:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"{category_name}_{timestamp}.json")
                scraper.save_to_json(category_products, filename)
                
                total_products += len(category_products)
                logger.info(f"Completed scraping {len(category_products)} products from category {category_name}")
            else:
                logger.warning(f"No products scraped from category {category_name}")
                
        except Exception as e:
            logger.error(f"Error processing category {category_name}: {str(e)}")
    
    # Calculate and log execution time
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Scraping completed. Total products scraped: {total_products}")
    logger.info(f"Total execution time: {duration}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        # Print the full traceback for debugging
        import traceback
        logger.critical(traceback.format_exc())