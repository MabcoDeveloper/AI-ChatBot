# test_mongo_atlas.py
from pymongo import MongoClient
from datetime import datetime
import sys

# Your Atlas connection string
MONGO_URI = "mongodb+srv://gradpro11223344:userone@cluster0.lomqiss.mongodb.net/?appName=Cluster0"

try:
    print("Testing MongoDB Atlas connection...")
    
    # Connect to MongoDB Atlas
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
    
    # List databases
    databases = client.list_database_names()
    print(f"📊 Available databases: {databases}")
    
    # Use beauty_bot database
    db = client['test']
    
    # Create or get collections
    products = db['products']
    offers = db['offers']
    
    # Test insert
    test_product = {
        "product_id": "TEST001",
        "name": "منتج اختبار",
        "price": 100.0,
        "category": "اختبار",
        "created_at": datetime.now()
    }
    
    result = products.insert_one(test_product)
    print(f"✅ Test document inserted with ID: {result.inserted_id}")
    
    # Count documents
    count = products.count_documents({})
    print(f"📈 Total products in database: {count}")
    
    # Clean up test document
    products.delete_one({"product_id": "TEST001"})
    print("🧹 Test document cleaned up")
    
    client.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check your internet connection")
    print("2. Verify your Atlas username/password")
    print("3. Make sure your IP is whitelisted in Atlas:")
    print("   - Go to Atlas Dashboard → Network Access")
    print("   - Add your current IP address (0.0.0.0/0 for all IPs)")
    print("4. Check if the cluster is running")
    sys.exit(1)