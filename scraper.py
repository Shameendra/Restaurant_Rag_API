"""
Web Scraper Module for Restaurant Data
Scrapes restaurant information, menus, and ratings from various sources
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from config import (
    USER_AGENT, 
    REQUEST_DELAY, 
    MAX_RESTAURANTS_TO_SCRAPE,
    SERPAPI_KEY,
    YELP_API_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Restaurant:
    """Data class representing a restaurant"""
    name: str
    address: str
    city: str
    rating: float
    review_count: int
    cuisine_type: str
    menu_items: List[str]
    price_range: str
    phone: str = ""
    website: str = ""
    source: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_text(self) -> str:
        """Convert restaurant data to text for embedding"""
        menu_text = ", ".join(self.menu_items) if self.menu_items else "Menu not available"
        return f"""
Restaurant: {self.name}
Location: {self.address}, {self.city}
Rating: {self.rating}/5 ({self.review_count} reviews)
Cuisine: {self.cuisine_type}
Price Range: {self.price_range}
Menu Items: {menu_text}
Phone: {self.phone}
Website: {self.website}
"""


class BaseScraper(ABC):
    """Abstract base class for restaurant scrapers"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
    
    @abstractmethod
    def search_restaurants(self, city: str, dish: str = None) -> List[Restaurant]:
        """Search for restaurants in a city"""
        pass
    
    def _delay(self):
        """Add delay between requests"""
        time.sleep(REQUEST_DELAY)
    
    def _safe_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a safe HTTP request with error handling"""
        try:
            self._delay()
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None


class YelpScraper(BaseScraper):
    """Scraper for Yelp restaurant data using Yelp Fusion API"""
    
    BASE_URL = "https://api.yelp.com/v3"
    
    def __init__(self):
        super().__init__()
        self.session.headers.update({
            "Authorization": f"Bearer {YELP_API_KEY}"
        })
    
    def search_restaurants(self, city: str, dish: str = None) -> List[Restaurant]:
        """Search restaurants on Yelp"""
        restaurants = []
        
        search_term = f"restaurants {dish}" if dish else "restaurants"
        
        url = f"{self.BASE_URL}/businesses/search"
        params = {
            "location": city,
            "term": search_term,
            "categories": "restaurants",
            "limit": MAX_RESTAURANTS_TO_SCRAPE,
            "sort_by": "rating"
        }
        
        response = self._safe_request(url, params)
        if not response:
            return restaurants
        
        try:
            data = response.json()
            businesses = data.get("businesses", [])
            
            for biz in businesses:
                # Get detailed business info including menu if available
                details = self._get_business_details(biz.get("id"))
                menu_items = details.get("menu_items", []) if details else []
                
                restaurant = Restaurant(
                    name=biz.get("name", "Unknown"),
                    address=" ".join(biz.get("location", {}).get("display_address", [])),
                    city=city,
                    rating=biz.get("rating", 0.0),
                    review_count=biz.get("review_count", 0),
                    cuisine_type=", ".join([c.get("title", "") for c in biz.get("categories", [])]),
                    menu_items=menu_items,
                    price_range=biz.get("price", "$$"),
                    phone=biz.get("phone", ""),
                    website=biz.get("url", ""),
                    source="Yelp"
                )
                restaurants.append(restaurant)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Yelp response: {e}")
        
        return restaurants
    
    def _get_business_details(self, business_id: str) -> Optional[Dict]:
        """Get detailed business information"""
        if not business_id:
            return None
            
        url = f"{self.BASE_URL}/businesses/{business_id}"
        response = self._safe_request(url)
        
        if response:
            try:
                return response.json()
            except json.JSONDecodeError:
                return None
        return None


class GooglePlacesScraper(BaseScraper):
    """Scraper using SerpAPI for Google search results"""
    
    BASE_URL = "https://serpapi.com/search"
    
    def search_restaurants(self, city: str, dish: str = None) -> List[Restaurant]:
        """Search restaurants using Google via SerpAPI"""
        restaurants = []
        
        query = f"best restaurants serving {dish} in {city}" if dish else f"best restaurants in {city}"
        
        params = {
            "engine": "google_local",
            "q": query,
            "location": city,
            "api_key": SERPAPI_KEY
        }
        
        response = self._safe_request(self.BASE_URL, params)
        if not response:
            return restaurants
        
        try:
            data = response.json()
            local_results = data.get("local_results", [])
            
            for result in local_results[:MAX_RESTAURANTS_TO_SCRAPE]:
                restaurant = Restaurant(
                    name=result.get("title", "Unknown"),
                    address=result.get("address", ""),
                    city=city,
                    rating=float(result.get("rating", 0)),
                    review_count=int(result.get("reviews", 0)),
                    cuisine_type=result.get("type", "Restaurant"),
                    menu_items=self._extract_menu_items(result),
                    price_range=result.get("price", "$$"),
                    phone=result.get("phone", ""),
                    website=result.get("website", ""),
                    source="Google"
                )
                restaurants.append(restaurant)
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Google response: {e}")
        
        return restaurants
    
    def _extract_menu_items(self, result: Dict) -> List[str]:
        """Extract menu items from search result snippets"""
        menu_items = []
        
        # Try to extract from extensions or snippets
        extensions = result.get("extensions", [])
        if isinstance(extensions, list):
            menu_items.extend([ext for ext in extensions if isinstance(ext, str)])
        
        return menu_items


class GenericWebScraper(BaseScraper):
    """Generic web scraper for restaurant websites"""
    
    def search_restaurants(self, city: str, dish: str = None) -> List[Restaurant]:
        """
        Generic search - scrapes from multiple sources
        This is a fallback when API-based scrapers are not available
        """
        restaurants = []
        
        # Search query
        query = f"restaurants+{dish}+{city}" if dish else f"restaurants+{city}"
        search_url = f"https://www.google.com/search?q={query}"
        
        response = self._safe_request(search_url)
        if not response:
            return restaurants
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse search results (simplified)
        # Note: Google's structure changes frequently
        for result in soup.select('.g')[:MAX_RESTAURANTS_TO_SCRAPE]:
            try:
                title_elem = result.select_one('h3')
                link_elem = result.select_one('a')
                snippet_elem = result.select_one('.VwiC3b')
                
                if title_elem and link_elem:
                    name = title_elem.get_text()
                    website = link_elem.get('href', '')
                    snippet = snippet_elem.get_text() if snippet_elem else ""
                    
                    # Extract rating if present in snippet
                    rating_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|/5)', snippet, re.I)
                    rating = float(rating_match.group(1)) if rating_match else 0.0
                    
                    restaurant = Restaurant(
                        name=name,
                        address="",
                        city=city,
                        rating=rating,
                        review_count=0,
                        cuisine_type="Restaurant",
                        menu_items=self._extract_menu_from_snippet(snippet),
                        price_range="$$",
                        website=website,
                        source="Web Search"
                    )
                    restaurants.append(restaurant)
                    
            except Exception as e:
                logger.debug(f"Failed to parse result: {e}")
                continue
        
        return restaurants
    
    def _extract_menu_from_snippet(self, snippet: str) -> List[str]:
        """Extract potential menu items from text snippet"""
        # Simple extraction - look for food-related words
        food_patterns = [
            r'\b(pizza|burger|pasta|salad|steak|sushi|tacos?|curry|soup|sandwich|wings?)\b',
            r'\b(chicken|beef|pork|fish|seafood|vegetarian|vegan)\b'
        ]
        
        items = []
        for pattern in food_patterns:
            matches = re.findall(pattern, snippet, re.I)
            items.extend(matches)
        
        return list(set(items))
    
    def scrape_menu_from_url(self, url: str) -> List[str]:
        """Attempt to scrape menu items from a restaurant's website"""
        menu_items = []
        
        response = self._safe_request(url)
        if not response:
            return menu_items
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for menu-related sections
        menu_selectors = [
            '.menu', '#menu', '[class*="menu"]',
            '.food-item', '.dish', '.menu-item',
            '[class*="dish"]', '[class*="food"]'
        ]
        
        for selector in menu_selectors:
            items = soup.select(selector)
            for item in items:
                text = item.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 100:
                    menu_items.append(text)
        
        # Fallback: look for common food words in the page
        if not menu_items:
            page_text = soup.get_text()
            menu_items = self._extract_menu_from_snippet(page_text)
        
        return list(set(menu_items))[:50]  # Limit to 50 items


class RestaurantDataCollector:
    """Main class to collect restaurant data from multiple sources"""
    
    def __init__(self):
        self.scrapers: List[BaseScraper] = []
        
        # Initialize available scrapers based on API keys
        if YELP_API_KEY and YELP_API_KEY != "your-yelp-api-key":
            self.scrapers.append(YelpScraper())
            logger.info("Yelp scraper initialized")
            
        if SERPAPI_KEY and SERPAPI_KEY != "your-serpapi-key":
            self.scrapers.append(GooglePlacesScraper())
            logger.info("Google Places scraper initialized")
        
        # Always include generic scraper as fallback
        self.scrapers.append(GenericWebScraper())
        logger.info("Generic web scraper initialized")
    
    def collect_restaurants(self, city: str, dish: str = None) -> List[Restaurant]:
        """Collect restaurants from all available sources"""
        all_restaurants = []
        seen_names = set()
        
        for scraper in self.scrapers:
            try:
                restaurants = scraper.search_restaurants(city, dish)
                
                # Deduplicate by restaurant name
                for restaurant in restaurants:
                    name_key = restaurant.name.lower().strip()
                    if name_key not in seen_names:
                        seen_names.add(name_key)
                        all_restaurants.append(restaurant)
                        
            except Exception as e:
                logger.error(f"Scraper {scraper.__class__.__name__} failed: {e}")
                continue
        
        # Sort by rating (descending)
        all_restaurants.sort(key=lambda x: (x.rating, x.review_count), reverse=True)
        
        return all_restaurants
    
    def collect_and_save(self, city: str, dish: str = None, output_file: str = None) -> str:
        """Collect restaurants and save to JSON file"""
        restaurants = self.collect_restaurants(city, dish)
        
        if not output_file:
            safe_city = city.lower().replace(" ", "_")
            safe_dish = dish.lower().replace(" ", "_") if dish else "all"
            output_file = f"restaurants_{safe_city}_{safe_dish}.json"
        
        data = {
            "city": city,
            "dish_query": dish,
            "restaurant_count": len(restaurants),
            "restaurants": [r.to_dict() for r in restaurants]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(restaurants)} restaurants to {output_file}")
        return output_file


# Demo/Test function
def demo_scraping():
    """Demo function to test scraping"""
    collector = RestaurantDataCollector()
    
    # Example: Find pizza restaurants in New York
    restaurants = collector.collect_restaurants("New York", "pizza")
    
    print(f"\nFound {len(restaurants)} restaurants:\n")
    for i, r in enumerate(restaurants[:5], 1):
        print(f"{i}. {r.name}")
        print(f"   Rating: {r.rating}/5 ({r.review_count} reviews)")
        print(f"   Address: {r.address}")
        print(f"   Cuisine: {r.cuisine_type}")
        print()


if __name__ == "__main__":
    demo_scraping()
