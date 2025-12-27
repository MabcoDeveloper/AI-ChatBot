#!/usr/bin/env python3
"""
Arabic Beauty Chatbot Runner
Run with: python run.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
import uvicorn
from config import settings

def check_dependencies():
    """Check if required services are running"""
    import socket
    import subprocess
    
    print("🔍 Checking dependencies...")
    
    # Check MongoDB
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 27017))
        if result == 0:
            print("✅ MongoDB is running")
        else:
            print("❌ MongoDB is not running. Starting MongoDB...")
            try:
                # Try to start MongoDB (Linux/Mac)
                subprocess.Popen(["mongod", "--fork", "--logpath", "/tmp/mongod.log"])
                print("✅ MongoDB started")
            except:
                print("⚠️  Please start MongoDB manually: mongod")
                print("   On Ubuntu: sudo systemctl start mongodb")
                print("   On Mac: brew services start mongodb-community")
    except Exception as e:
        print(f"⚠️  MongoDB check failed: {e}")
    
    # Check Redis
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 6379))
        if result == 0:
            print("✅ Redis is running")
        else:
            print("❌ Redis is not running. Starting Redis...")
            try:
                # Try to start Redis
                subprocess.Popen(["redis-server"])
                print("✅ Redis started")
            except:
                print("⚠️  Please start Redis manually:")
                print("   On Ubuntu: sudo systemctl start redis")
                print("   On Mac: brew services start redis")
    except Exception as e:
        print(f"⚠️  Redis check failed: {e}")

def initialize_database():
    """Initialize database with sample data"""
    print("📦 Initializing database...")
    try:
        from data.seed_products import seed_products
        result = seed_products()
        print(f"✅ Database initialized: {result['products_inserted']} products, {result['offers_inserted']} offers")
        return True
    except Exception as e:
        print(f"⚠️  Database initialization failed: {e}")
        print("   Trying alternative method...")
        try:
            from data.init_database import initialize
            initialize()
            print("✅ Database initialized with alternative method")
            return True
        except Exception as e2:
            print(f"❌ Failed to initialize database: {e2}")
            return False

def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("Arabic Beauty Products Chatbot")
    print("="*50)
    
    # Check dependencies
    if len(sys.argv) > 1 and sys.argv[1] != "--skip-checks":
        check_dependencies()
    
    # Initialize database
    if len(sys.argv) > 1 and sys.argv[1] != "--skip-init":
        initialize_database()
    
    # Run the application
    print("\n🚀 Starting chatbot API server...")
    print(f"   📍 Local: http://{settings.APP_HOST}:{settings.APP_PORT}")
    print(f"   📍 Docs: http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
    print(f"   📍 Health: http://{settings.APP_HOST}:{settings.APP_PORT}/health")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    main()