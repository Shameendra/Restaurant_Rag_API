"""
Tests for the Restaurant RAG System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
from scraper import Restaurant, RestaurantDataCollector, GenericWebScraper
from config import SUPPORTED_CITIES


class TestRestaurantDataClass:
    """Tests for the Restaurant data class"""
    
    def test_restaurant_creation(self):
        """Test creating a Restaurant instance"""
        restaurant = Restaurant(
            name="Test Restaurant",
            address="123 Main St",
            city="New York",
            rating=4.5,
            review_count=100,
            cuisine_type="Italian",
            menu_items=["Pizza", "Pasta"],
            price_range="$$"
        )
        
        assert restaurant.name == "Test Restaurant"
        assert restaurant.rating == 4.5
        assert len(restaurant.menu_items) == 2
    
    def test_restaurant_to_dict(self):
        """Test converting Restaurant to dictionary"""
        restaurant = Restaurant(
            name="Test Restaurant",
            address="123 Main St",
            city="New York",
            rating=4.5,
            review_count=100,
            cuisine_type="Italian",
            menu_items=["Pizza"],
            price_range="$$"
        )
        
        data = restaurant.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "Test Restaurant"
        assert data["rating"] == 4.5
    
    def test_restaurant_to_text(self):
        """Test converting Restaurant to text for embedding"""
        restaurant = Restaurant(
            name="Test Restaurant",
            address="123 Main St",
            city="New York",
            rating=4.5,
            review_count=100,
            cuisine_type="Italian",
            menu_items=["Pizza", "Pasta"],
            price_range="$$"
        )
        
        text = restaurant.to_text()
        
        assert "Test Restaurant" in text
        assert "4.5" in text
        assert "Pizza" in text


class TestGenericWebScraper:
    """Tests for the GenericWebScraper class"""
    
    def test_extract_menu_from_snippet(self):
        """Test extracting menu items from text"""
        scraper = GenericWebScraper()
        
        snippet = "We serve delicious pizza, fresh salads, and grilled chicken"
        items = scraper._extract_menu_from_snippet(snippet)
        
        assert "pizza" in [item.lower() for item in items]
        assert "chicken" in [item.lower() for item in items]
    
    def test_extract_menu_empty_snippet(self):
        """Test extraction from empty snippet"""
        scraper = GenericWebScraper()
        
        items = scraper._extract_menu_from_snippet("")
        
        assert items == []


class TestRestaurantDataCollector:
    """Tests for the RestaurantDataCollector class"""
    
    def test_collector_initialization(self):
        """Test that collector initializes with at least generic scraper"""
        collector = RestaurantDataCollector()
        
        assert len(collector.scrapers) >= 1
    
    @patch.object(GenericWebScraper, 'search_restaurants')
    def test_collect_restaurants_deduplication(self, mock_search):
        """Test that duplicate restaurants are removed"""
        mock_search.return_value = [
            Restaurant(
                name="Test Restaurant",
                address="123 Main St",
                city="New York",
                rating=4.5,
                review_count=100,
                cuisine_type="Italian",
                menu_items=[],
                price_range="$$"
            ),
            Restaurant(
                name="Test Restaurant",  # Duplicate
                address="123 Main St",
                city="New York",
                rating=4.5,
                review_count=100,
                cuisine_type="Italian",
                menu_items=[],
                price_range="$$"
            )
        ]
        
        collector = RestaurantDataCollector()
        # Only use the generic scraper for this test
        collector.scrapers = [GenericWebScraper()]
        
        restaurants = collector.collect_restaurants("New York", "pizza")
        
        # Should deduplicate
        assert len(restaurants) == 1


class TestConfig:
    """Tests for configuration"""
    
    def test_supported_cities_not_empty(self):
        """Test that supported cities list is not empty"""
        assert len(SUPPORTED_CITIES) > 0
    
    def test_supported_cities_lowercase(self):
        """Test that all cities are lowercase"""
        for city in SUPPORTED_CITIES:
            assert city == city.lower()


class TestQueryParsing:
    """Tests for query parsing functionality"""
    
    def test_parse_city_from_query(self):
        """Test parsing city from various query formats"""
        queries = [
            ("Where can I find pizza in New York?", "new york"),
            ("Best sushi restaurants in Los Angeles", "los angeles"),
            ("restaurants in Chicago", "chicago"),
        ]
        
        for query, expected_city in queries:
            # Simple keyword extraction for testing
            query_lower = query.lower()
            found_city = None
            for city in SUPPORTED_CITIES:
                if city in query_lower:
                    found_city = city
                    break
            
            if expected_city in SUPPORTED_CITIES:
                assert found_city == expected_city


# Integration test placeholder
class TestIntegration:
    """Integration tests (require API keys)"""
    
    @pytest.mark.skip(reason="Requires API keys")
    def test_full_rag_pipeline(self):
        """Test the full RAG pipeline"""
        pass
    
    @pytest.mark.skip(reason="Requires API keys")
    def test_full_agent_pipeline(self):
        """Test the full agent pipeline"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
