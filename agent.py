"""
LangGraph Agent for Restaurant Discovery
Implements a stateful agent workflow for complex restaurant queries
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from dataclasses import dataclass
import operator
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, TOP_K_RESULTS
from scraper import RestaurantDataCollector, Restaurant
from rag_engine import RestaurantVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Define the agent state
class AgentState(TypedDict):
    """State for the restaurant agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    city: str
    dish: str
    restaurants: List[Dict]
    search_completed: bool
    final_response: str


# Define tools for the agent
@tool
def search_restaurants(city: str, dish: str = None) -> str:
    """
    Search for restaurants in a specific city, optionally filtering by dish.
    
    Args:
        city: The city to search in (e.g., "New York", "Los Angeles")
        dish: Optional specific dish to search for (e.g., "pizza", "sushi")
    
    Returns:
        JSON string with restaurant information including names, ratings, and menu items
    """
    collector = RestaurantDataCollector()
    restaurants = collector.collect_restaurants(city, dish)
    
    if not restaurants:
        return f"No restaurants found in {city}" + (f" serving {dish}" if dish else "")
    
    # Format results
    results = []
    for r in restaurants[:10]:  # Limit to top 10
        results.append({
            "name": r.name,
            "rating": r.rating,
            "review_count": r.review_count,
            "address": r.address,
            "cuisine": r.cuisine_type,
            "price_range": r.price_range,
            "menu_items": r.menu_items[:10] if r.menu_items else []
        })
    
    return str(results)


@tool
def get_restaurant_details(restaurant_name: str, city: str) -> str:
    """
    Get detailed information about a specific restaurant.
    
    Args:
        restaurant_name: Name of the restaurant
        city: City where the restaurant is located
    
    Returns:
        Detailed information about the restaurant
    """
    collector = RestaurantDataCollector()
    restaurants = collector.collect_restaurants(city)
    
    # Find matching restaurant
    for r in restaurants:
        if restaurant_name.lower() in r.name.lower():
            return f"""
Restaurant: {r.name}
Location: {r.address}, {r.city}
Rating: {r.rating}/5.0 ({r.review_count} reviews)
Cuisine: {r.cuisine_type}
Price Range: {r.price_range}
Phone: {r.phone}
Website: {r.website}
Menu Items: {', '.join(r.menu_items[:15]) if r.menu_items else 'Not available'}
"""
    
    return f"Restaurant '{restaurant_name}' not found in {city}"


@tool  
def compare_restaurants(restaurant_names: List[str], city: str) -> str:
    """
    Compare multiple restaurants side by side.
    
    Args:
        restaurant_names: List of restaurant names to compare
        city: City where restaurants are located
    
    Returns:
        Comparison table of restaurants
    """
    collector = RestaurantDataCollector()
    restaurants = collector.collect_restaurants(city)
    
    comparisons = []
    for name in restaurant_names:
        for r in restaurants:
            if name.lower() in r.name.lower():
                comparisons.append({
                    "name": r.name,
                    "rating": r.rating,
                    "reviews": r.review_count,
                    "price": r.price_range,
                    "cuisine": r.cuisine_type
                })
                break
    
    if not comparisons:
        return "No matching restaurants found for comparison"
    
    # Format as comparison
    result = "Restaurant Comparison:\n" + "=" * 50 + "\n"
    for c in comparisons:
        result += f"\n{c['name']}\n"
        result += f"  Rating: {c['rating']}/5 ({c['reviews']} reviews)\n"
        result += f"  Price: {c['price']}\n"
        result += f"  Cuisine: {c['cuisine']}\n"
    
    return result


class RestaurantAgent:
    """LangGraph-based agent for restaurant discovery"""
    
    SYSTEM_PROMPT = """You are a helpful restaurant recommendation assistant.
Your goal is to help users find the best restaurants serving specific dishes in their desired city.

You have access to the following tools:
- search_restaurants: Search for restaurants in a city, optionally by dish
- get_restaurant_details: Get detailed info about a specific restaurant
- compare_restaurants: Compare multiple restaurants

When responding:
1. First understand what the user is looking for (city, dish type)
2. Use tools to gather restaurant information
3. Present the top 5 recommendations with ratings and relevant details
4. Be helpful and provide actionable information

Always include restaurant ratings and review counts in your recommendations."""

    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
        self.tools = [search_restaurants, get_restaurant_details, compare_restaurants]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_query", self._parse_query_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("respond", self._respond_node)
        
        # Add edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "search")
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def _parse_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Parse the user query to extract city and dish"""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the city and dish from the query.
Respond in exactly this format:
CITY: <city name>
DISH: <dish name or 'any' if not specified>"""),
            ("human", "{query}")
        ])
        
        chain = parse_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": last_message})
        
        # Parse result
        city = ""
        dish = ""
        
        for line in result.split("\n"):
            if "CITY:" in line:
                city = line.split("CITY:")[1].strip()
            elif "DISH:" in line:
                dish = line.split("DISH:")[1].strip()
                if dish.lower() == "any":
                    dish = ""
        
        logger.info(f"Parsed query - City: {city}, Dish: {dish}")
        
        return {
            "city": city,
            "dish": dish
        }
    
    def _search_node(self, state: AgentState) -> Dict[str, Any]:
        """Search for restaurants"""
        city = state.get("city", "")
        dish = state.get("dish", "")
        
        if not city:
            return {
                "restaurants": [],
                "search_completed": True,
                "messages": [AIMessage(content="I couldn't identify the city. Please specify which city you're looking for restaurants in.")]
            }
        
        # Use the search tool
        collector = RestaurantDataCollector()
        restaurants = collector.collect_restaurants(city, dish if dish else None)
        
        restaurant_dicts = [r.to_dict() for r in restaurants[:10]]
        
        logger.info(f"Found {len(restaurant_dicts)} restaurants")
        
        return {
            "restaurants": restaurant_dicts,
            "search_completed": True
        }
    
    def _analyze_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyze and rank restaurants"""
        restaurants = state.get("restaurants", [])
        
        if not restaurants:
            return {"restaurants": []}
        
        # Sort by rating and review count
        sorted_restaurants = sorted(
            restaurants,
            key=lambda x: (x.get("rating", 0), x.get("review_count", 0)),
            reverse=True
        )
        
        return {"restaurants": sorted_restaurants[:TOP_K_RESULTS]}
    
    def _respond_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate final response"""
        city = state.get("city", "")
        dish = state.get("dish", "")
        restaurants = state.get("restaurants", [])
        
        if not restaurants:
            response = f"I couldn't find any restaurants"
            if dish:
                response += f" serving {dish}"
            if city:
                response += f" in {city}"
            response += ". Please try a different search."
            
            return {
                "final_response": response,
                "messages": [AIMessage(content=response)]
            }
        
        # Build response with LLM
        context = self._format_restaurants_context(restaurants)
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Based on the following restaurant data, provide the top {k} recommendations 
for finding {dish} in {city}. Include ratings, review counts, and any relevant menu items.

Restaurant Data:
{context}

Please format your response clearly with numbered recommendations.""")
        ])
        
        chain = response_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "k": TOP_K_RESULTS,
            "dish": dish if dish else "food",
            "city": city,
            "context": context
        })
        
        return {
            "final_response": response,
            "messages": [AIMessage(content=response)]
        }
    
    def _format_restaurants_context(self, restaurants: List[Dict]) -> str:
        """Format restaurants for LLM context"""
        parts = []
        for i, r in enumerate(restaurants, 1):
            menu_items = r.get("menu_items", [])
            menu_str = ", ".join(menu_items[:10]) if menu_items else "Not available"
            
            parts.append(f"""
Restaurant {i}: {r.get('name', 'Unknown')}
- Rating: {r.get('rating', 'N/A')}/5.0 ({r.get('review_count', 0)} reviews)
- Address: {r.get('address', 'N/A')}, {r.get('city', 'N/A')}
- Cuisine: {r.get('cuisine_type', 'N/A')}
- Price Range: {r.get('price_range', 'N/A')}
- Menu Items: {menu_str}
""")
        
        return "\n".join(parts)
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and return restaurant recommendations
        
        Args:
            user_query: Natural language query like "Where can I find tacos in Austin?"
        
        Returns:
            Restaurant recommendations
        """
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "city": "",
            "dish": "",
            "restaurants": [],
            "search_completed": False,
            "final_response": ""
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state.get("final_response", "Unable to process your request.")
    
    def chat(self, user_query: str, history: List[BaseMessage] = None) -> str:
        """
        Chat interface with conversation history
        
        Args:
            user_query: User's message
            history: Previous conversation messages
        
        Returns:
            Agent response
        """
        messages = history or []
        messages.append(HumanMessage(content=user_query))
        
        initial_state = {
            "messages": messages,
            "city": "",
            "dish": "",
            "restaurants": [],
            "search_completed": False,
            "final_response": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return final_state.get("final_response", "Unable to process your request.")


# Simple interface function
def ask_restaurant_agent(query: str) -> str:
    """
    Simple function to query the restaurant agent
    
    Args:
        query: Question like "Where can I find sushi in Seattle?"
    
    Returns:
        Restaurant recommendations
    """
    agent = RestaurantAgent()
    return agent.query(query)


# Demo function
def demo_agent():
    """Demo the LangGraph agent"""
    agent = RestaurantAgent()
    
    queries = [
        "Where can I find the best pizza in New York City?",
        "I'm looking for sushi restaurants in Los Angeles",
        "What are the top-rated Mexican restaurants in Austin?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        response = agent.query(query)
        print(response)
        print()


if __name__ == "__main__":
    demo_agent()
