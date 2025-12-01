"""
FastAPI Web Server for Restaurant RAG System
Provides REST API endpoints for restaurant discovery
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging

from rag_engine import RestaurantRAG
from agent import RestaurantAgent
from scraper import RestaurantDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant RAG API",
    description="Find restaurants serving specific dishes in any city using RAG + LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines (lazy loading)
_rag_engine: Optional[RestaurantRAG] = None
_agent_engine: Optional[RestaurantAgent] = None


def get_rag_engine() -> RestaurantRAG:
    """Get or create RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RestaurantRAG()
    return _rag_engine


def get_agent_engine() -> RestaurantAgent:
    """Get or create Agent engine singleton"""
    global _agent_engine
    if _agent_engine is None:
        _agent_engine = RestaurantAgent()
    return _agent_engine


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for natural language queries"""
    query: str = Field(..., description="Natural language query", example="Where can I find pizza in New York?")
    use_agent: bool = Field(default=True, description="Use LangGraph agent (True) or simple RAG (False)")
    refresh_data: bool = Field(default=True, description="Fetch fresh restaurant data")


class SearchRequest(BaseModel):
    """Request model for structured search"""
    city: str = Field(..., description="City to search in", example="New York")
    dish: Optional[str] = Field(None, description="Dish or cuisine to search for", example="pizza")
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20)


class RestaurantResponse(BaseModel):
    """Response model for restaurant data"""
    name: str
    address: str
    city: str
    rating: float
    review_count: int
    cuisine_type: str
    price_range: str
    menu_items: List[str]


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str
    response: str
    restaurants_found: int


class SearchResponse(BaseModel):
    """Response model for search results"""
    city: str
    dish: Optional[str]
    count: int
    restaurants: List[RestaurantResponse]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/query", response_model=QueryResponse)
async def query_restaurants(request: QueryRequest):
    """
    Process a natural language query about restaurants
    
    Examples:
    - "Where can I find pizza in New York?"
    - "Best sushi restaurants in Los Angeles"
    - "Top rated Mexican food in Austin"
    """
    try:
        if request.use_agent:
            engine = get_agent_engine()
            response = engine.query(request.query)
        else:
            engine = get_rag_engine()
            response = engine.query(request.query, refresh=request.refresh_data)
        
        return QueryResponse(
            query=request.query,
            response=response,
            restaurants_found=5  # Approximate
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_restaurants(request: SearchRequest):
    """
    Search for restaurants by city and optional dish
    
    Returns structured restaurant data
    """
    try:
        collector = RestaurantDataCollector()
        restaurants = collector.collect_restaurants(request.city, request.dish)
        
        # Convert to response format
        restaurant_responses = []
        for r in restaurants[:request.top_k]:
            restaurant_responses.append(RestaurantResponse(
                name=r.name,
                address=r.address,
                city=r.city,
                rating=r.rating,
                review_count=r.review_count,
                cuisine_type=r.cuisine_type,
                price_range=r.price_range,
                menu_items=r.menu_items[:10] if r.menu_items else []
            ))
        
        return SearchResponse(
            city=request.city,
            dish=request.dish,
            count=len(restaurant_responses),
            restaurants=restaurant_responses
        )
        
    except Exception as e:
        logger.error(f"Error searching restaurants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/{city}")
async def search_by_city(
    city: str,
    dish: Optional[str] = Query(None, description="Dish to search for"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results")
):
    """
    Search restaurants by city (GET endpoint)
    
    Example: /search/New%20York?dish=pizza&top_k=5
    """
    request = SearchRequest(city=city, dish=dish, top_k=top_k)
    return await search_restaurants(request)


@app.get("/ask")
async def ask_question(
    q: str = Query(..., description="Your question about restaurants"),
    agent: bool = Query(True, description="Use agent mode")
):
    """
    Simple GET endpoint for asking questions
    
    Example: /ask?q=Where+can+I+find+pizza+in+NYC
    """
    request = QueryRequest(query=q, use_agent=agent)
    return await query_restaurants(request)


# Run server
def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
