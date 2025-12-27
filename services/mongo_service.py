from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import sys
import re
import difflib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBService:
    """MongoDB service for beauty products data using MongoDB Atlas"""
    
    def __init__(self, mongo_uri: str = None):
        # Your Atlas connection string
        self.mongo_uri = mongo_uri or "mongodb+srv://yaseen:test123@cluster0.p99gzup.mongodb.net/?retryWrites=true&w=majority"
        self.db_name = "beauty_bot"
        
        try:
            logger.info("Connecting to MongoDB Atlas...")
            
            # Connect to MongoDB Atlas with optimized settings
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=10000,  # 10 seconds timeout
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                maxPoolSize=50,
                retryWrites=True,
                appname="ArabicBeautyBot"
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            
            # Collections
            self.products: Collection = self.db['products']
            self.offers: Collection = self.db['offers']
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB Atlas database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {e}")
            print(f"\n❌ Error: {e}")
            print("\nPlease check:")
            print("1. Your internet connection")
            print("2. MongoDB Atlas cluster status")
            print("3. IP whitelist in Atlas Network Access")
            print("4. Username and password")
            sys.exit(1)
    
    def _create_indexes(self):
        """Create necessary indexes for performance - FIXED for MongoDB compatibility"""
        try:
            # Products indexes - FIXED: removed default_language parameter
            if "name_text" not in self.products.index_information():
                # Don't specify default_language for TEXT index
                self.products.create_index([("name", TEXT)], name="name_text")
            
            # Regular indexes
            indexes_to_create = [
                ("name_asc", [("name", ASCENDING)]),
                ("category_asc", [("category", ASCENDING)]),
                ("brand_asc", [("brand", ASCENDING)]),
                ("price_asc", [("price", ASCENDING)]),
                ("in_stock_asc", [("in_stock", ASCENDING)]),
                ("created_at_desc", [("created_at", DESCENDING)])
            ]
            
            for index_name, fields in indexes_to_create:
                if index_name not in self.products.index_information():
                    self.products.create_index(fields, name=index_name)
            
            logger.info("✅ MongoDB indexes created/verified")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")
    
    def search_products(self, query: str = None, category: str = None, 
                       brand: str = None, limit: int = 20, min_price: float = None, max_price: float = None,
                       sort_by: Optional[str] = None, sort_order: Optional[int] = None) -> List[Dict]:
        """Search products with filters, price range and optional sorting"""
        
        filter_query = {}
        
        if query and query.strip():
            # Use text search for better results
            filter_query["$text"] = {"$search": query}
        
        if category and category.strip():
            escaped_cat = re.escape(category.strip())
            # Partial, case-insensitive match
            filter_query["category"] = {"$regex": escaped_cat, "$options": "i"}
        
        if brand and brand.strip():
            escaped_brand = re.escape(brand.strip())
            # Partial, case-insensitive match
            filter_query["brand"] = {"$regex": escaped_brand, "$options": "i"}

        # Price range filters
        if min_price is not None or max_price is not None:
            price_filter = {}
            if min_price is not None:
                price_filter["$gte"] = float(min_price)
            if max_price is not None:
                price_filter["$lte"] = float(max_price)
            filter_query["price"] = price_filter
        
        # Execute query
        cursor = self.products.find(
            filter_query,
            {
                "_id": 0,
                "product_id": 1,
                "name": 1,
                "description": 1,
                "price": 1,
                "original_price": 1,
                "currency": 1,
                "category": 1,
                "brand": 1,
                "in_stock": 1,
                "stock_quantity": 1,
                "attributes": 1,
                "image_url": 1,
                "rating": 1,
                "review_count": 1
            }
        )

        # Apply sorting if requested
        if sort_by == 'price':
            order = ASCENDING if sort_order == 1 else DESCENDING
            cursor = cursor.sort('price', order)

        cursor = cursor.limit(limit)
        
        products = list(cursor)

        # Fallback fuzzy matching when text search yields no results (improves spelling tolerance)
        if (not products or len(products) == 0) and query and query.strip():
            try:
                # Lightweight normalization for Arabic to improve matching
                def _simplify_text(s: str) -> str:
                    if not s:
                        return ""
                    v = s.lower()
                    mappings = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه', 'ڤ': 'ف'}
                    for a, b in mappings.items():
                        v = v.replace(a, b)
                    v = re.sub(r'[^0-9\u0600-\u06FF\s]', '', v)
                    v = re.sub(r'\s+', ' ', v).strip()
                    return v

                simplified_query = _simplify_text(query)

                # Collect product names and normalized forms (small collections ok)
                names = self.products.distinct("name")
                name_pairs = [(n, _simplify_text(n or "")) for n in names if n]
                candidates = {orig: norm for orig, norm in name_pairs}

                if candidates:
                    # Use difflib to get close matches on simplified strings
                    matches = difflib.get_close_matches(simplified_query, list(candidates.values()), n=limit, cutoff=0.6)
                    matched_originals = [orig for orig, norm in candidates.items() if norm in matches]

                    if matched_originals:
                        cursor2 = self.products.find(
                            {"name": {"$in": matched_originals}},
                            {
                                "_id": 0,
                                "product_id": 1,
                                "name": 1,
                                "description": 1,
                                "price": 1,
                                "original_price": 1,
                                "currency": 1,
                                "category": 1,
                                "brand": 1,
                                "in_stock": 1,
                                "stock_quantity": 1,
                                "attributes": 1,
                                "image_url": 1,
                                "rating": 1,
                                "review_count": 1
                            }
                        )
                        if sort_by == 'price':
                            order = ASCENDING if sort_order == 1 else DESCENDING
                            cursor2 = cursor2.sort('price', order)
                        cursor2 = cursor2.limit(limit)
                        products = list(cursor2)
            except Exception:
                # If fuzzy fallback fails, return the empty result we already have
                pass

        return products
    
    def get_product_by_name(self, product_name: str) -> Optional[Dict]:
        """Get product by name (fuzzy match)"""
        product = self.products.find_one(
            {"name": {"$regex": product_name, "$options": "i"}},
            {"_id": 0}
        )
        return product
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product by ID"""
        product = self.products.find_one(
            {"product_id": product_id},
            {"_id": 0}
        )
        return product
    
    def get_current_offers(self, limit: int = 10) -> List[Dict]:
        """Get current active offers"""
        now = datetime.now()
        
        cursor = self.offers.find(
            {
                "active": True,
                "starts_at": {"$lte": now},
                "$or": [
                    {"expires_at": {"$gte": now}},
                    {"expires_at": None}
                ]
            },
            {"_id": 0}
        ).limit(limit)
        
        return list(cursor)
    
    def seed_sample_data(self):
        """Seed sample beauty products data"""
        from datetime import datetime
        
        sample_products = [
            {
                "product_id": "BP001",
                "name": "كريم مرطب للوجه من نيڤيا",
                "description": "كريم مرطب يومي للوجه يناسب جميع أنواع البشرة",
                "price": 45.99,
                "original_price": 59.99,
                "currency": "SAR",
                "category": "العناية بالبشرة",
                "brand": "نيڤيا",
                "in_stock": True,
                "stock_quantity": 150,
                "attributes": {"skin_type": ["جميع أنواع البشرة"], "volume_ml": 100},
                "image_url": "https://example.com/nivea.jpg",
                "rating": 4.5,
                "review_count": 128,
                "created_at": datetime.now()
            },
            {
                "product_id": "BP002",
                "name": "شامبو للشعر الجاف من بانين",
                "description": "شامبو مغذي للشعر الجاف والمتقصف",
                "price": 32.50,
                "currency": "SAR",
                "category": "العناية بالشعر",
                "brand": "بانين",
                "in_stock": True,
                "stock_quantity": 85,
                "attributes": {"hair_type": ["جاف", "متقصف"], "volume_ml": 400},
                "rating": 4.4,
                "review_count": 142,
                "created_at": datetime.now()
            },
            {
                "product_id": "BP003",
                "name": "أحمر شفاه مات من ماك",
                "description": "أحمر شفاه بتشطيب مات، لون كلاسيكي أحمر",
                "price": 89.00,
                "original_price": 110.00,
                "currency": "SAR",
                "category": "مكياج",
                "brand": "ماك",
                "in_stock": True,
                "stock_quantity": 42,
                "attributes": {"color": "أحمر كلاسيكي", "finish": "مات"},
                "rating": 4.8,
                "review_count": 312,
                "created_at": datetime.now()
            },
            {
                "product_id": "BP004",
                "name": "عطر فلورال من شانيل",
                "description": "عطر نسائي برائحة زهور الربيع",
                "price": 350.00,
                "original_price": 420.00,
                "currency": "SAR",
                "category": "العطور",
                "brand": "شانيل",
                "in_stock": True,
                "stock_quantity": 23,
                "attributes": {"fragrance_type": "فلورال", "volume_ml": 50},
                "rating": 4.9,
                "review_count": 267,
                "created_at": datetime.now()
            },
            {
                "product_id": "BP005",
                "name": "سيروم فيتامين سي",
                "description": "سيروم مضاد للأكسدة يضيء البشرة",
                "price": 120.00,
                "original_price": 150.00,
                "currency": "SAR",
                "category": "العناية بالبشرة",
                "brand": "ذا أورديناري",
                "in_stock": False,
                "stock_quantity": 0,
                "attributes": {"skin_type": ["جميع أنواع البشرة"], "vitamin_c_percentage": 15},
                "rating": 4.7,
                "review_count": 256,
                "created_at": datetime.now()
            }
        ]
        
        sample_offers = [
            {
                "offer_id": "OFF001",
                "title": "تخفيضات العناية بالبشرة",
                "description": "خصم 20% على جميع منتجات العناية بالبشرة",
                "discount_percentage": 20,
                "category": "العناية بالبشرة",
                "starts_at": datetime.now() - timedelta(days=1),
                "expires_at": datetime.now() + timedelta(days=7),
                "active": True,
                "created_at": datetime.now()
            },
            {
                "offer_id": "OFF002",
                "title": "عرض العطور الفاخرة",
                "description": "خصم 15% على جميع العطور الأصلية",
                "discount_percentage": 15,
                "category": "العطور",
                "starts_at": datetime.now() - timedelta(days=2),
                "expires_at": datetime.now() + timedelta(days=5),
                "active": True,
                "created_at": datetime.now()
            }
        ]
        
        try:
            # Clear existing data
            self.products.delete_many({})
            self.offers.delete_many({})
            
            # Insert sample data
            self.products.insert_many(sample_products)
            self.offers.insert_many(sample_offers)
            
            logger.info(f"✅ Seeded {len(sample_products)} products and {len(sample_offers)} offers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to seed data: {e}")
            return False
    
    def count_products(self) -> int:
        """Count total products in database"""
        return self.products.count_documents({})
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Create singleton instance
mongo_service = MongoDBService()