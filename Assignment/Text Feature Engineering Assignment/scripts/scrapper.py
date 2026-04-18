import csv
import time
import re
import os
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class FlipkartScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_driver(self):
        options = uc.ChromeOptions()
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        return uc.Chrome(options=options, use_subprocess=True)

    def get_top_reviews(self, product_url, count=5):
        driver = self._get_driver()
        reviews = []
        try:
            # Smart URL Transformation: Always go to the full reviews page
            review_url = product_url.replace("/p/", "/product-reviews/")
            driver.get(review_url)
            time.sleep(5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            # Combined selectors for text blocks
            review_blocks = soup.select("div.Z_37_N, div.EPCmJX, div._6K-7Co")
            
            seen = set()
            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True).replace("READ MORE", "").strip()
                if len(text) > 20 and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count: break
        finally:
            driver.quit()
        return " || ".join(reviews) if reviews else "No reviews found"

    def scrape_flipkart_products(self, query, max_products=1, review_count=5):
        driver = self._get_driver()
        wait = WebDriverWait(driver, 15)
        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        
        products = []
        try:
            driver.get(search_url)
            
            # --- FIX: NO MORE FRAGILE CLASSES ---
            # Instead of looking for a class, we look for ANY link that contains '/p/'
            product_xpath = "(//a[contains(@href, '/p/')])[1]"
            product_el = wait.until(EC.presence_of_element_located((By.XPATH, product_xpath)))
            
            href = product_el.get_attribute("href")
            title = "Product Result" # Default if title classes fail
            
            # Try to grab title from common classes if available
            try:
                title = driver.find_element(By.CSS_SELECTOR, "div.KzDlHZ, div._3wU53n, div._4rR01T").text
            except: pass

            print(f"✅ Found Product: {title}")
            driver.quit() # Close search driver to save memory

            # Now fetch reviews using the URL we just found
            top_reviews = self.get_top_reviews(href, count=review_count)
            products.append(["N/A", title, "N/A", "N/A", "N/A", top_reviews])

        except Exception as e:
            print(f"❌ Search Error: {e}")
            driver.quit()

        return products

    def save_to_csv(self, data, query):
        df = pd.DataFrame(data, columns=["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
        filename = f"{query.replace(' ', '_').lower()}_results.csv"
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"📂 Saved data to {path}")

if __name__ == "__main__":
    query = "iphone 16"
    scraper = FlipkartScraper()
    data = scraper.scrape_flipkart_products(query)
    if data:
        scraper.save_to_csv(data, query)