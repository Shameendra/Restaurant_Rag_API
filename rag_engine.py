"""
RAG (Retrieval Augmented Generation) Module
Handles document embedding, vector storage, and retrieval
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.schema import Document
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS
)
from scraper import Restaurant, RestaurantDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class RestaurantVectorStore:
    """Vector store for restaurant data with FAISS"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        self.vector_store: Optional[FAISS] = None
        self.restaurants: Dict[str, Restaurant] = {}
    
    def add_restaurants(self, restaurants: List[Restaurant]) -> None:
        """Add restaurants to the vector store"""
        documents = []
        
        for restaurant in restaurants:
            # Store restaurant for later retrieval
            self.restaurants[restaurant.name] = restaurant
            
            # Create document with restaurant text
            text = restaurant.to_text()
            
            # Create metadata for filtering and display
            metadata = {
                "name": restaurant.name,
                "city": restaurant.city,
                "rating": restaurant.rating,
                "review_count": restaurant.review_count,
                "cuisine_type": restaurant.cuisine_type,
                "price_range": restaurant.price_range,
                "source": restaurant.source
            }
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        # Split documents if needed
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vector_store.add_documents(split_docs)
        
        logger.info(f"Added {len(restaurants)} restaurants to vector store")
    
    def search(self, query: str, k: int = TOP_K_RESULTS, city_filter: str = None) -> List[Document]:
        """Search for relevant restaurants"""
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k * 2)  # Get more for filtering
        
        # Filter by city if specified
        if city_filter:
            city_lower = city_filter.lower()
            results = [
                doc for doc in results 
                if doc.metadata.get("city", "").lower() == city_lower
            ]
        
        # Return top k results
        return results[:k]
    
    def search_with_scores(self, query: str, k: int = TOP_K_RESULTS) -> List[tuple]:
        """Search with relevance scores"""
        if self.vector_store is None:
            return []
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def save(self, path: str = "restaurant_vectorstore") -> None:
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            
            # Save restaurant data
            restaurant_data = {
                name: r.to_dict() for name, r in self.restaurants.items()
            }
            with open(f"{path}_restaurants.json", 'w') as f:
                json.dump(restaurant_data, f, indent=2)
            
            logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = "restaurant_vectorstore") -> None:
        """Load vector store from disk"""
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load restaurant data
        restaurant_file = f"{path}_restaurants.json"
        if os.path.exists(restaurant_file):
            with open(restaurant_file, 'r') as f:
                data = json.load(f)
                self.restaurants = {
                    name: Restaurant(**r) for name, r in data.items()
                }
        
        logger.info(f"Vector store loaded from {path}")


class RestaurantRAG:
    """RAG system for restaurant queries"""
    
    SYSTEM_PROMPT = """You are a helpful restaurant recommendation assistant. 
Your task is to help users find restaurants serving specific dishes in their desired city.

Based on the retrieved restaurant information, provide helpful and accurate recommendations.
Always include:
1. Restaurant name
2. Rating and number of reviews
3. Address/Location
4. Relevant menu items (if available)
5. Price range

If the information is not available in the context, say so honestly.
Format your response in a clear, easy-to-read manner."""

    USER_PROMPT_TEMPLATE = """
User Query: {query}

Retrieved Restaurant Information:
{context}

Based on the above information, please provide the top restaurant recommendations 
that best match the user's query. Include ratings and relevant details for each restaurant.
"""

    def __init__(self):
        self.vector_store = RestaurantVectorStore()
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE
        )
        self.collector = RestaurantDataCollector()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT_TEMPLATE)
        ])
        
        # Create chain
        self.chain = (
            {"context": self._retrieve, "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _retrieve(self, query: str) -> str:
        """Retrieve relevant documents and format as context"""
        docs = self.vector_store.search(query, k=TOP_K_RESULTS)
        
        if not docs:
            return "No restaurant information found for this query."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"--- Restaurant {i} ---\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def refresh_data(self, city: str, dish: str = None) -> int:
        """Scrape fresh restaurant data and add to vector store"""
        logger.info(f"Refreshing data for city: {city}, dish: {dish}")
        
        restaurants = self.collector.collect_restaurants(city, dish)
        
        if restaurants:
            self.vector_store.add_restaurants(restaurants)
            logger.info(f"Added {len(restaurants)} restaurants to knowledge base")
        
        return len(restaurants)
    
    def query(self, user_query: str, refresh: bool = True) -> str:
        """
        Process user query and return restaurant recommendations
        
        Args:
            user_query: Natural language query (e.g., "Where can I find pizza in New York?")
            refresh: Whether to scrape fresh data before answering
        
        Returns:
            AI-generated response with restaurant recommendations
        """
        # Parse city and dish from query
        city, dish = self._parse_query(user_query)
        
        if refresh and city:
            self.refresh_data(city, dish)
        
        # Run RAG chain
        response = self.chain.invoke(user_query)
        
        return response
    
    def _parse_query(self, query: str) -> tuple:
        """Parse city and dish from user query using LLM"""
        parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the city and dish from the user's query.
Return in format: CITY: <city name> | DISH: <dish name>
If city or dish is not mentioned, use "unknown" for that field.
Examples:
- "Where can I find pizza in New York?" -> CITY: New York | DISH: pizza
- "Best sushi restaurants in Tokyo" -> CITY: Tokyo | DISH: sushi
- "Restaurants in Chicago" -> CITY: Chicago | DISH: unknown"""),
            ("human", "{query}")
        ])
        
        parse_chain = parse_prompt | self.llm | StrOutputParser()
        result = parse_chain.invoke({"query": query})
        
        # Parse result
        city = "unknown"
        dish = None
        
        try:
            parts = result.split("|")
            for part in parts:
                if "CITY:" in part:
                    city = part.split("CITY:")[1].strip()
                elif "DISH:" in part:
                    dish_val = part.split("DISH:")[1].strip()
                    if dish_val.lower() != "unknown":
                        dish = dish_val
        except Exception as e:
            logger.warning(f"Failed to parse query: {e}")
        
        return city if city.lower() != "unknown" else None, dish
    
    def save_knowledge_base(self, path: str = "restaurant_kb") -> None:
        """Save the knowledge base to disk"""
        self.vector_store.save(path)
    
    def load_knowledge_base(self, path: str = "restaurant_kb") -> None:
        """Load knowledge base from disk"""
        self.vector_store.load(path)


# Convenience function for direct queries
def find_restaurants(query: str, refresh: bool = True) -> str:
    """
    Simple function to find restaurants based on query
    
    Args:
        query: Natural language query like "Where can I find tacos in Austin?"
        refresh: Whether to fetch fresh data
    
    Returns:
        Restaurant recommendations as formatted string
    """
    rag = RestaurantRAG()
    return rag.query(query, refresh=refresh)


# Demo function
def demo_rag():
    """Demo the RAG system"""
    rag = RestaurantRAG()
    
    # Example query
    query = "Where can I find the best pizza in New York City?"
    
    print(f"Query: {query}\n")
    print("Searching and analyzing restaurants...\n")
    
    response = rag.query(query)
    
    print("=" * 50)
    print("RECOMMENDATIONS:")
    print("=" * 50)
    print(response)


if __name__ == "__main__":
    demo_rag()
