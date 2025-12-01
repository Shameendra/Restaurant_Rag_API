# 🍽️ Restaurant RAG System

A Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph that helps users find restaurants serving specific dishes in any city. The system scrapes live restaurant data, stores it in a vector database, and uses an LLM to provide intelligent recommendations.

## 🎯 Features

- **Natural Language Queries**: Ask questions like "Where can I find pizza in New York?"
- **Live Data Scraping**: Fetches real-time restaurant data from multiple sources
- **RAG Architecture**: Uses vector embeddings for semantic search
- **LangGraph Agent**: Sophisticated multi-step reasoning workflow
- **Multiple Data Sources**: Supports Yelp API, Google Places (via SerpAPI), and web scraping
- **REST API**: FastAPI server for integration with other applications
- **Interactive CLI**: Chat-style command line interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│            "Where can I find sushi in Seattle?"              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query Parser (LLM)                         │
│              Extracts: city="Seattle", dish="sushi"          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Web Scraper Module                          │
│    ┌──────────┐  ┌──────────┐  ┌──────────────────┐        │
│    │   Yelp   │  │  Google  │  │  Generic Scraper │        │
│    │   API    │  │  Places  │  │   (Fallback)     │        │
│    └──────────┘  └──────────┘  └──────────────────┘        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vector Store (FAISS)                       │
│           Embeds restaurant data using OpenAI                │
│           Enables semantic similarity search                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              LangGraph Agent / RAG Chain                     │
│    Retrieves relevant restaurants & generates response       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Final Response                             │
│   "Here are the top 5 sushi restaurants in Seattle:         │
│    1. Sushi Kashiba - ⭐ 4.8 (2,341 reviews)..."            │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
restaurant-rag-project/
├── config.py          # Configuration settings and API keys
├── scraper.py         # Web scraping modules for restaurant data
├── rag_engine.py      # RAG system with vector store
├── agent.py           # LangGraph agent implementation
├── main.py            # CLI application entry point
├── api.py             # FastAPI REST server
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone or create the project directory
cd restaurant-rag-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Required:**
- `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

**Optional (for better results):**
- `SERPAPI_KEY`: Get from [SerpAPI](https://serpapi.com/)
- `YELP_API_KEY`: Get from [Yelp Developers](https://www.yelp.com/developers/v3/manage_app)

### 3. Run the Application

**Interactive Mode:**
```bash
python main.py --interactive
```

**Single Query:**
```bash
python main.py --query "Where can I find pizza in New York?"
```

**Structured Search:**
```bash
python main.py --city "Los Angeles" --dish "sushi"
```

**Start API Server:**
```bash
python api.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 💻 Usage Examples

### Python API

```python
from main import RestaurantFinder, quick_search

# Using the RestaurantFinder class
finder = RestaurantFinder()
result = finder.find("Where can I find tacos in Austin?")
print(result)

# Quick search function
result = quick_search("Chicago", "deep dish pizza")
print(result)
```

### REST API

**Query Endpoint:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Best sushi in Seattle", "use_agent": true}'
```

**Search Endpoint:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"city": "New York", "dish": "pizza", "top_k": 5}'
```

**Simple GET Request:**
```bash
curl "http://localhost:8000/ask?q=Where+can+I+find+ramen+in+Tokyo"
```

### LangGraph Agent Direct Usage

```python
from agent import RestaurantAgent

agent = RestaurantAgent()

# Single query
response = agent.query("Where can I find authentic Italian food in Boston?")
print(response)

# Chat with history
from langchain_core.messages import HumanMessage, AIMessage

history = []
response1 = agent.chat("Find pizza places in NYC", history)
history.append(HumanMessage(content="Find pizza places in NYC"))
history.append(AIMessage(content=response1))

response2 = agent.chat("Which one has the best reviews?", history)
print(response2)
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o" for better results
TEMPERATURE = 0.3

# Search Settings
TOP_K_RESULTS = 5
MAX_RESTAURANTS_TO_SCRAPE = 20

# Vector Store Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

## 📊 How It Works

### 1. Query Parsing
The system uses an LLM to extract the city and dish from natural language queries:
- Input: "Where can I find the best pizza in New York City?"
- Output: `city="New York City"`, `dish="pizza"`

### 2. Data Collection
Multiple scrapers fetch restaurant data:
- **Yelp API**: Ratings, reviews, business details
- **Google Places**: Location data, popular dishes
- **Generic Scraper**: Fallback web scraping

### 3. Vector Embedding
Restaurant data is converted to text and embedded:
```
Restaurant: Joe's Pizza
Location: 7 Carmine St, New York
Rating: 4.5/5 (3,241 reviews)
Cuisine: Pizza, Italian
Menu Items: Margherita, Pepperoni, Cheese Slice...
```

### 4. Semantic Search
User queries are embedded and matched against restaurant vectors using FAISS for efficient similarity search.

### 5. Response Generation
The LLM generates a natural language response based on retrieved restaurant data, including:
- Restaurant names and rankings
- Ratings and review counts
- Addresses and contact info
- Relevant menu items

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 🐳 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "api.py"]
```

```bash
# Build and run
docker build -t restaurant-rag .
docker run -p 8000:8000 --env-file .env restaurant-rag
```

## 🔐 Security Notes

- Never commit `.env` files with real API keys
- Use environment variables in production
- Rate limit API endpoints in production
- Respect website robots.txt when scraping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📝 License

MIT License - feel free to use this project for learning and development.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Stateful agent workflows
- [OpenAI](https://openai.com/) - Embeddings and LLM
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

## 📧 Support

For issues or questions, please open a GitHub issue or contact the maintainers.

---

**Happy Restaurant Hunting! 🍕🍣🌮🍔**
