"""
Restaurant RAG Application - Main Entry Point
Find restaurants serving specific dishes in any city using RAG + LLM
"""

import argparse
import sys
from typing import Optional
import logging

from rag_engine import RestaurantRAG, find_restaurants
from agent import RestaurantAgent, ask_restaurant_agent
from scraper import RestaurantDataCollector
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RestaurantFinder:
    """Main application class for restaurant discovery"""
    
    def __init__(self, use_agent: bool = True):
        """
        Initialize the restaurant finder
        
        Args:
            use_agent: Use LangGraph agent (True) or simple RAG (False)
        """
        self.use_agent = use_agent
        
        if use_agent:
            self.engine = RestaurantAgent()
            logger.info("Initialized LangGraph Agent")
        else:
            self.engine = RestaurantRAG()
            logger.info("Initialized RAG Engine")
    
    def find(self, query: str) -> str:
        """
        Find restaurants based on natural language query
        
        Args:
            query: Natural language query like "Where can I find pizza in NYC?"
        
        Returns:
            Formatted restaurant recommendations
        """
        logger.info(f"Processing query: {query}")
        
        if self.use_agent:
            return self.engine.query(query)
        else:
            return self.engine.query(query, refresh=True)
    
    def find_by_params(self, city: str, dish: str) -> str:
        """
        Find restaurants by specific parameters
        
        Args:
            city: City name
            dish: Dish/cuisine to search for
        
        Returns:
            Formatted restaurant recommendations
        """
        query = f"Where can I find {dish} in {city}?"
        return self.find(query)
    
    def interactive_mode(self):
        """Run in interactive chat mode"""
        print("\n" + "="*60)
        print("🍽️  Restaurant Finder - Interactive Mode")
        print("="*60)
        print("\nAsk me about restaurants! Examples:")
        print("  - Where can I find pizza in New York?")
        print("  - Best sushi restaurants in Los Angeles")
        print("  - Top rated tacos in Austin")
        print("\nType 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 🍽️")
                    break
                
                print("\n🔍 Searching...\n")
                response = self.find(query)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 🍽️")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\nSorry, I encountered an error. Please try again.\n")


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Find restaurants serving specific dishes in any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py --interactive
  
  # Single query
  python main.py --query "Where can I find pizza in New York?"
  
  # Search by city and dish
  python main.py --city "Los Angeles" --dish "sushi"
  
  # Use simple RAG instead of agent
  python main.py --query "Best tacos in Austin" --no-agent
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Natural language query (e.g., "Where can I find pizza in NYC?")'
    )
    
    parser.add_argument(
        '--city', '-c',
        type=str,
        help='City to search in'
    )
    
    parser.add_argument(
        '--dish', '-d',
        type=str,
        help='Dish or cuisine to search for'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive chat mode'
    )
    
    parser.add_argument(
        '--no-agent',
        action='store_true',
        help='Use simple RAG instead of LangGraph agent'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize finder
    use_agent = not args.no_agent
    finder = RestaurantFinder(use_agent=use_agent)
    
    # Handle different modes
    if args.interactive:
        finder.interactive_mode()
    
    elif args.query:
        print(f"\n🔍 Searching: {args.query}\n")
        result = finder.find(args.query)
        print(result)
    
    elif args.city and args.dish:
        print(f"\n🔍 Searching for {args.dish} in {args.city}\n")
        result = finder.find_by_params(args.city, args.dish)
        print(result)
    
    elif args.city:
        print(f"\n🔍 Searching for restaurants in {args.city}\n")
        query = f"What are the best restaurants in {args.city}?"
        result = finder.find(query)
        print(result)
    
    else:
        # Default to interactive mode if no arguments
        finder.interactive_mode()


# Quick access functions for programmatic use
def quick_search(city: str, dish: str) -> str:
    """
    Quick search function for programmatic use
    
    Args:
        city: City name (e.g., "New York")
        dish: Dish to search for (e.g., "pizza")
    
    Returns:
        Restaurant recommendations as string
    """
    finder = RestaurantFinder()
    return finder.find_by_params(city, dish)


def query_restaurants(query: str) -> str:
    """
    Process a natural language query
    
    Args:
        query: Natural language query
    
    Returns:
        Restaurant recommendations
    """
    finder = RestaurantFinder()
    return finder.find(query)


if __name__ == "__main__":
    main()
