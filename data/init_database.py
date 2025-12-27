#!/usr/bin/env python3
"""
Initialize database (alternative method)
Run with: python -m data.init_database
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.mongo_service import MongoDBService
from datetime import datetime, timedelta

def initialize():
    """Initialize database with minimal data"""
    
    print("🔧 Initializing database...")
    
    try:
        # Create MongoDB service
        mongo_service = MongoDBService()
        
        # Check if database is already populated
        product_count = mongo_service.products.count_documents({})
        offer_count = mongo_service.offers.count_documents({})
        
        if product_count > 0:
            print(f"✅ Database already has {product_count} products and {offer_count} offers")
            return True
        
        # Import seed function
        from data.seed_products import seed_products
        return seed_products()
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        
        # Try to create minimal data
        try:
            print("🔄 Creating minimal dataset...")
            mongo_service = MongoDBService()
            
            # Add one sample product
            sample_product = {
                "product_id": "SAMPLE001",
                "name": "كريم مرطب عينة",
                "description": "كريم مرطب للوجه",
                "price": 29.99,
                "currency": "SAR",
                "category": "العناية بالبشرة",
                "brand": "عينة",
                "in_stock": True,
                "stock_quantity": 100,
                "created_at": datetime.now()
            }
            
            mongo_service.products.insert_one(sample_product)
            print("✅ Added sample product")
            return True
            
        except Exception as e2:
            print(f"❌ Could not create minimal dataset: {e2}")
            return False

if __name__ == "__main__":
    initialize()