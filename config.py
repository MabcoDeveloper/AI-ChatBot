from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # MongoDB Atlas
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb+srv://gradpro11223344:userone@cluster0.lomqiss.mongodb.net/?appName=Cluster0")
    MONGO_DB: str = os.getenv("MONGO_DB", "test")
    MONGO_PRODUCTS_COLLECTION: str = "products"
    MONGO_OFFERS_COLLECTION: str = "offers"
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TTL: int = 3600
    
    # Application
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True
    
    # Intent Patterns (Arabic)
    SEARCH_PATTERNS: List[str] = [
        "ابحث عن", "عندكم", "وين", "أين", "بتحتوي على", "محتاج", "بدي", "اريد",
        "شو في", "ما في", "متوفر", "موجود"
    ]
    PRICE_PATTERNS: List[str] = [
        "سعر", "كم السعر", "بكم", "التكلفة", "الثمن", "الكمية", "الخصم",
        "البيع", "كم بيبلش"
    ]
    OFFER_PATTERNS: List[str] = [
        "عروض", "خصومات", "تخفيضات", "عرض", "تنزيلات", "عروض اليوم",
        "سوبرسيل", "تخفيض"
    ]
    HELP_PATTERNS: List[str] = [
        "مساعدة", "مساعدة عامة", "شو بتقدر", "شغلات", "وظائف", "ممكن",
        "تقدر تساعدني", "كيف", "طريقة"
    ]

settings = Settings()