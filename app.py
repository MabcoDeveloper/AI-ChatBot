#!/usr/bin/env python3
"""
Arabic Beauty Chatbot API - Fixed Version
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from datetime import datetime
import logging
import time

# Import services
from services.mongo_service import mongo_service
from services.chatbot_service import chatbot_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arabic Beauty Chatbot API",
    description="Arabic-first e-commerce conversational agent for beauty products",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for your UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models matching your UI
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User message")
    timestamp: Optional[str] = Field(None, description="Timestamp from client")
    # session_id is optional and will be ignored if present

class ChatResponse(BaseModel):
    user_id: str
    original_message: str
    normalized_message: str
    intent: str
    intent_confidence: float
    response: str
    data: Optional[Dict[str, Any]]
    suggestions: List[str]
    context_summary: Dict[str, Any]
    timestamp: str
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    products_count: int
    offers_count: int


class CartAddRequest(BaseModel):
    product_id: str
    quantity: Optional[int] = 1
    cart_id: Optional[str] = None
    customer_name: Optional[str] = None
    phone: Optional[str] = None


# Track startup time
startup_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("🚀 Starting Arabic Beauty Chatbot API")
    
    # Seed database if empty
    try:
        count = mongo_service.count_products()
        if count == 0:
            logger.info("Database is empty, seeding sample data...")
            mongo_service.seed_sample_data()
        else:
            logger.info(f"✅ Database has {count} products")
    except Exception as e:
        logger.warning(f"⚠️ Could not seed database: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "مرحباً! أنا مساعد التجميل العربي",
        "status": "running",
        "database": "MongoDB Atlas",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "products": "GET /products",
            "offers": "GET /offers",
            "seed": "POST /seed (for testing)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        mongo_service.client.admin.command('ping')
        
        # Get counts
        products_count = mongo_service.count_products()
        offers_count = mongo_service.offers.count_documents({})
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "products_count": products_count,
            "offers_count": offers_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, response: Response, background_tasks: BackgroundTasks = None):
    """
    Main chat endpoint - FIXED to work with your UI
    
    Your UI sends: { user_id, message, timestamp }
    We ignore session_data if present
    This endpoint will also set a cookie if chatbot returns `data['set_cookie']`.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request from user {request.user_id}: {request.message[:50]}...")
        
        # Process message through chatbot service
        # Note: We don't pass session_data since your chatbot_service doesn't expect it
        result = chatbot_service.process_message(
            user_id=request.user_id,
            message=request.message
        )
        
        # If the chatbot included set_cookie directives, set them on the response
        try:
            sc = (result.get('data') or {}).get('set_cookie')
            if sc and isinstance(sc, dict):
                cookie_name = sc.get('name', 'cart_id')
                cookie_value = sc.get('value')
                max_age = sc.get('max_age', 30*24*3600)
                # set HTTP-only cookie so UI JavaScript cannot read it if not necessary
                response.set_cookie(key=cookie_name, value=cookie_value, max_age=max_age, httponly=True)
        except Exception as e:
            logger.debug(f"Failed to set cookie from chatbot result: {e}")
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(processing_time_ms, 2)
        
        logger.info(f"Chat response generated: intent={result['intent']}, time={processing_time_ms:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        # Return a fallback response instead of raising an error
        return {
            "user_id": request.user_id,
            "original_message": request.message,
            "normalized_message": request.message,
            "intent": "error",
            "intent_confidence": 0.0,
            "response": "عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى.",
            "data": None,
            "suggestions": ["جرب مرة أخرى", "اتصل بالدعم"],
            "context_summary": {
                "turns_count": 0,
                "last_activity": None,
                "user_messages": 0,
                "bot_messages": 0,
                "last_intent": None
            },
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }

@app.post("/clear")
async def clear_conversation(user_id: str):
    """Clear conversation history for a user"""
    success = chatbot_service.clear_conversation(user_id)
    return {
        "success": success,
        "message": f"Conversation cleared for user {user_id}" if success else "User not found",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/products")
async def get_products(search: Optional[str] = None, category: Optional[str] = None, brand: Optional[str] = None, limit: int = 20):
    """Get products from database. Supports optional filters: category and brand."""
    products = mongo_service.search_products(query=search, category=category, brand=brand, limit=limit)
    return {
        "products": products,
        "count": len(products),
        "search_query": search,
        "category": category,
        "brand": brand,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/offers")
async def get_offers():
    """Get current offers"""
    offers = mongo_service.get_current_offers()
    return {
        "offers": offers,
        "count": len(offers),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/seed")
async def seed_database():
    """Seed database with sample data (for testing)"""
    success = mongo_service.seed_sample_data()
    return {
        "success": success,
        "message": "Database seeded with sample data" if success else "Failed to seed database",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/cart/add")
async def add_to_cart(req: CartAddRequest, response: Response):
    """Add a product to a shopping cart. Accepts either cart_id or (customer_name + phone). Returns updated cart and sets a cookie with `cart_id`."""
    try:
        if req.cart_id:
            cart = mongo_service.get_cart_by_id(req.cart_id)
            if not cart:
                raise HTTPException(status_code=404, detail="Cart not found")
        elif req.customer_name and req.phone:
            cart = mongo_service.get_or_create_cart_by_customer(req.customer_name, req.phone)
        else:
            raise HTTPException(status_code=400, detail="Provide cart_id or customer_name and phone")

        updated = mongo_service.add_item_to_cart(cart['cart_id'], req.product_id, req.quantity)
        response.set_cookie(key='cart_id', value=updated['cart_id'], max_age=30*24*3600, httponly=True)
        return {"success": True, "cart": updated}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add to cart failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cart")
async def get_cart(cart_id: Optional[str] = None, customer_name: Optional[str] = None, phone: Optional[str] = None):
    """Get cart by cart_id or by customer_name + phone"""
    if cart_id:
        cart = mongo_service.get_cart_by_id(cart_id)
    elif customer_name and phone:
        cart = mongo_service.get_cart_by_customer(customer_name, phone)
    else:
        raise HTTPException(status_code=400, detail="Provide cart_id or customer_name and phone")
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")
    return {"cart": cart}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "status": "ok",
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            {"method": "GET", "path": "/", "description": "Root endpoint"},
            {"method": "POST", "path": "/chat", "description": "Chat with bot"},
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "GET", "path": "/products", "description": "Get products"},
            {"method": "GET", "path": "/offers", "description": "Get offers"}
        ]
    }

# Error handler
@app.exception_handler(Exception)
async def universal_exception_handler(request, exc):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again.",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Arabic Beauty Chatbot API")
    print("="*50)
    print("📡 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🧪 Test: http://localhost:8000/test")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )