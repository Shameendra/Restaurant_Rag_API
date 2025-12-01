"""
Configuration settings for the Restaurant RAG System
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your-serpapi-key")  # For Google search
YELP_API_KEY = os.getenv("YELP_API_KEY", "your-yelp-api-key")  # Optional

# LLM Settings
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o" for better results
TEMPERATURE = 0.3
MAX_TOKENS = 2000

# Vector Store Settings
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Scraping Settings
MAX_RESTAURANTS_TO_SCRAPE = 20
REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Search Settings
TOP_K_RESULTS = 5

# Supported Cities (can be extended)
SUPPORTED_CITIES = [
    "new york", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "fort worth", "columbus", "charlotte",
    "seattle", "denver", "boston", "detroit", "nashville",
    "london", "paris", "berlin", "tokyo", "sydney"
]
