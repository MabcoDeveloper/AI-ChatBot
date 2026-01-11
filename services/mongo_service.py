from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.collection import Collection
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import sys
import re
import difflib
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InMemoryCursor:
    """Simple cursor wrapper to support sort and limit chainable calls for tests."""
    def __init__(self, items):
        self._items = list(items)
    def sort(self, key, order):
        reverse = (order == -1)
        try:
            self._items.sort(key=lambda x: x.get(key) if x.get(key) is not None else 0, reverse=reverse)
        except Exception:
            pass
        return self
    def limit(self, n):
        self._items = self._items[:n]
        return self
    def __iter__(self):
        return iter(self._items)
    def __len__(self):
        return len(self._items)
    def __repr__(self):
        return repr(self._items)

class InMemoryCollection:
    """A tiny in-memory replacement for the minimal MongoDB features used by the app.
    Supports find, find_one, distinct, insert_many, delete_many, insert_one, update_one, delete_one, count_documents.
    """
    def __init__(self, initial=None):
        self._docs = list(initial) if initial else []
    def find(self, filter_query=None, projection=None):
        filter_query = filter_query or {}
        def match(doc):
            # $text: check in name or description
            if '$text' in filter_query:
                q = filter_query['$text'].get('$search') if isinstance(filter_query['$text'], dict) else filter_query['$text']
                q = q or ''
                if q.lower() not in (doc.get('name','') + ' ' + doc.get('description','')).lower():
                    return False
            for k, v in filter_query.items():
                if k == '$text':
                    continue
                if isinstance(v, dict) and '$regex' in v:
                    try:
                        if not re.search(v['$regex'], doc.get(k,''), flags=re.I):
                            return False
                    except Exception:
                        return False
                elif k == 'price' and isinstance(v, dict):
                    pv = doc.get('price') or 0
                    if '$gte' in v and pv < v['$gte']:
                        return False
                    if '$lte' in v and pv > v['$lte']:
                        return False
                else:
                    # direct equality or other simple checks
                    if k in doc and isinstance(v, (str,int,float)):
                        if str(v).lower() not in str(doc.get(k,'')).lower():
                            return False
            return True
        matched = [d.copy() for d in self._docs if match(d)]
        return InMemoryCursor(matched)
    def distinct(self, field):
        vals = []
        for d in self._docs:
            v = d.get(field)
            if v and v not in vals:
                vals.append(v)
        return vals
    def find_one(self, query, projection=None):
        c = self.find(query)
        for d in c:
            return d
        return None
    def insert_many(self, docs):
        for d in docs:
            self._docs.append(dict(d))
    def delete_many(self, query=None):
        if not query or query == {}:
            removed = len(self._docs)
            self._docs = []
            class Res: deleted_count = removed
            return Res()
        before = len(self._docs)
        self._docs = [d for d in self._docs if not all(d.get(k) == v for k, v in query.items())]
        class Res: deleted_count = before - len(self._docs)
        return Res()
    def insert_one(self, doc):
        self._docs.append(dict(doc))
        class Res: inserted_id = True
        return Res()
    def update_one(self, query, update):
        for d in self._docs:
            match = True
            for k, v in query.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                if '$push' in update:
                    for k, v in update['$push'].items():
                        d.setdefault(k, []).append(v)
                if '$set' in update:
                    for k, v in update['$set'].items():
                        d[k] = v
                class Res: modified_count = 1
                return Res()
        class Res: modified_count = 0
        return Res()
    def delete_one(self, query):
        for i, d in enumerate(self._docs):
            ok = True
            for k, v in query.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                del self._docs[i]
                class Res: deleted_count = 1
                return Res()
        class Res: deleted_count = 0
        return Res()
    def count_documents(self, query=None):
        if not query or query == {}:
            return len(self._docs)
        cnt = 0
        for d in self._docs:
            if all(d.get(k) == v for k, v in query.items()):
                cnt += 1
        return cnt
    def index_information(self):
        return {}
    def create_index(self, *args, **kwargs):
        return True

class MongoDBService:
    """MongoDB service for beauty products data using MongoDB Atlas"""
    
    def __init__(self, mongo_uri: str = None):
        # Your Atlas connection string
        self.mongo_uri = mongo_uri or "mongodb+srv://gradpro11223344:userone@cluster0.lomqiss.mongodb.net/?appName=Cluster0"
        self.db_name = "test"
        
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
            # Carts collection to store customer shopping carts
            self.carts: Collection = self.db['carts']
            # Collection to store conversation logs / training examples
            self.training_examples: Collection = self.db['training_examples']
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB Atlas database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {e}")
            logger.warning("Falling back to in-memory database for local testing (no network or Atlas unreachable).")
            # Initialize in-memory collections so the app can run without an Atlas connection
            self.client = None
            self.db = None
            self.products = InMemoryCollection()
            self.offers = InMemoryCollection()
            self.carts = InMemoryCollection()
            # Create indexes no-op (handled by InMemoryCollection)
            logger.info("✅ In-memory database initialized")
            # Attempt to seed sample data to make the service useful locally
            try:
                self.seed_sample_data()
            except Exception as se:
                logger.warning(f"Seeding in-memory data failed: {se}")

    
    def _create_indexes(self):
        """Create necessary indexes for performance - FIXED for MongoDB compatibility"""
        try:
            # Products text index across English and Arabic fields for broad text search
            idx_info = self.products.index_information()
            # Detect any existing text indexes so we can migrate to the new weighted 'full_text' index
            text_idx_names = []
            for iname, info in (idx_info or {}).items():
                try:
                    key = info.get('key', [])
                    for k in key:
                        if isinstance(k, (list, tuple)) and len(k) >= 2 and k[1] == 'text':
                            text_idx_names.append(iname)
                            break
                except Exception:
                    continue

            # If an older text index exists (different name/options), drop it to avoid IndexOptionsConflict
            for ti in list(set(text_idx_names)):
                if ti != 'full_text':
                    try:
                        # Some drivers/in-memory collections may not support drop_index; ignore errors
                        self.products.drop_index(ti)
                        logger.info("Dropped existing text index: %s", ti)
                    except Exception as e:
                        logger.warning("Could not drop index %s: %s", ti, e)

            # Create the new full_text index if it's not present
            idx_info = self.products.index_information()
            if "full_text" not in (idx_info or {}):
                try:
                    self.products.create_index([
                        ("name", TEXT), ("title", TEXT), ("description", TEXT),
                        ("title_ar", TEXT), ("description_ar", TEXT)
                    ], name="full_text", weights={"name": 10, "title": 8, "title_ar": 8, "description": 4, "description_ar": 4}, default_language='english')
                    logger.info("Created text index 'full_text' for English/Arabic fields")
                except Exception as e:
                    logger.warning(f"Could not create 'full_text' index: {e}")
            
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

            # Carts indexes
            try:
                carts_indexes = self.carts.index_information()
                if "cart_id_idx" not in carts_indexes:
                    self.carts.create_index([("cart_id", ASCENDING)], name="cart_id_idx", unique=True)
                if "customer_phone_idx" not in carts_indexes:
                    self.carts.create_index([("customer.phone", ASCENDING)], name="customer_phone_idx")
            except Exception as e:
                logger.debug(f"Could not create carts indexes: {e}")


            logger.info("✅ MongoDB indexes created/verified")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create indexes: {e}")
    

    # ------------------- Training data helpers -------------------
    def insert_training_session(self, session_doc: Dict[str, Any]) -> Optional[str]:
        """Create a new training session document and return its id (or session_id field)."""
        try:
            if self.training_examples is None:
                return None
            res = self.training_examples.insert_one(session_doc)
            try:
                return str(res.inserted_id)
            except Exception:
                return session_doc.get('session_id')
        except Exception as e:
            logger.warning(f"Failed to insert training session: {e}")
            return None

    def log_training_turn(self, session_id: str, turn: Dict[str, Any]) -> bool:
        """Append a turn to an existing training session (by session_id)."""
        try:
            if self.training_examples is None:
                return False
            q = {'session_id': session_id}
            update = {'$push': {'turns': turn}, '$set': {'last_updated': datetime.utcnow()}}
            res = self.training_examples.update_one(q, {'$push': {'turns': turn}, '$set': {'last_updated': datetime.utcnow()}}, upsert=True)
            return res.modified_count >= 0
        except Exception as e:
            logger.warning(f"Failed to log training turn: {e}")
            return False

    def finalize_training_session(self, session_id: str, outcome: str, summary: Dict[str, Any] = None) -> bool:
        """Mark a training session as finished with an outcome label and optional summary."""
        try:
            if self.training_examples is None:
                return False
            update_doc = {'outcome': outcome, 'ended_at': datetime.utcnow()}
            if summary:
                update_doc['summary'] = summary
            res = self.training_examples.update_one({'session_id': session_id}, {'$set': update_doc})
            return res.modified_count >= 0
        except Exception as e:
            logger.warning(f"Failed to finalize training session: {e}")
            return False

    def fetch_training_sessions(self, filter_query: Dict[str, Any] = None, limit: int = 100):
        try:
            q = filter_query or {}
            cursor = self.training_examples.find(q)
            return list(cursor)[:limit]
        except Exception as e:
            logger.warning(f"Failed to fetch training sessions: {e}")
            return []
    
    def search_products(self, query: str = None, category: str = None, 
                       brand: str = None, product_type: str = None, limit: int = 20, min_price: float = None, max_price: float = None,
                       sort_by: Optional[str] = None, sort_order: Optional[int] = None) -> List[Dict]:
        """Search products with filters, price range and optional sorting"""
        
        filter_query = {}
        
        # Reusable projection for product queries
        proj = {
            "_id": 0,
            "product_id": 1,
            "name": 1,
            "title": 1,
            "title_ar": 1,
            "description": 1,
            "description_ar": 1,
            "price": 1,
            "min_price": 1,
            "max_price": 1,
            "price_map": 1,
            "original_price": 1,
            "currency": 1,
            "category": 1,
            "brand": 1,
            "in_stock": 1,
            "stock_quantity": 1,
            "attributes": 1,
            "images": 1,
            "image_url": 1,
            "rating": 1,
            "review_count": 1
        }

        if query and query.strip():
            esc = re.escape(query)
            # If query contains Arabic characters, try a stricter token-AND match on the Arabic name/description
            if re.search(r'[\u0600-\u06FF]', query):
                # lightweight Arabic normalization for tokens and product fields
                def _simplify_ar(s: str) -> str:
                    if not s:
                        return ""
                    v = s.lower()
                    mappings = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': '', 'ڤ': 'ف'}
                    for a, b in mappings.items():
                        v = v.replace(a, b)
                    # remove punctuation and keep Arabic letters and numbers
                    v = re.sub(r'[^0-9\u0600-\u06FF\s]', '', v)
                    # remove definite article 'ال' at word starts (normalize 'الشعر' -> 'شعر')
                    v = re.sub(r'(^|\s)ال', r"\1", v)
                    v = re.sub(r'\s+', ' ', v).strip()
                    return v

                tokens_raw = [t for t in re.split(r'\s+', query) if t.strip()]
                # Normalize common possessives and endings (e.g., شعري -> شعر)
                def _normalize_token(t: str) -> str:
                    t = t.strip()
                    t = re.sub(r"^(?:my|ل)?","", t)  # remove leading possessive particles if present
                    t = re.sub(r"(ي|ها|نا|كم|هم)$", "", t)  # strip simple suffixes
                    return t

                # For Arabic queries, only use Arabic tokens and match against Arabic fields in DB
                tokens = [_simplify_ar(_normalize_token(t)) for t in tokens_raw if t.strip()]
                # Filter out common stopwords and very short tokens that cause noisy matches
                stopwords = {'من','في','على','عن','مع','اعاني','أعاني','اعانى','انا','مشكلة','عندي','لدي','هل','يا','ارجو','ممكن'}
                tokens = [t for t in tokens if t and len(t) > 2 and t not in stopwords]
                if tokens:
                    try:
                        token_regex = '|'.join([re.escape(tok) for tok in set(tokens) if tok])
                        # Search only Arabic title/description fields to find candidates
                        cand_cursor = self.products.find({
                            "$or": [
                                {"title_ar": {"$regex": token_regex, "$options": "i"}},
                                {"description_ar": {"$regex": token_regex, "$options": "i"}}
                            ]
                        }, proj).limit(400)
                        candidates = list(cand_cursor)

                        # Detect high-level domain tokens in query (skin vs hair) to resolve ambiguous tokens like 'جاف' appearing in both contexts
                        domain = None
                        domain_map = {
                            'skin': ['بشرت', 'بشرة', 'بشرتي', 'بشرتك', 'وجه', 'جلد'],
                            'hair': ['شعر', 'شعري', 'شعرك']
                        }
                        for t in tokens:
                            for dname, kws in domain_map.items():
                                if any(kw in t for kw in kws):
                                    domain = dname
                                    break
                            if domain:
                                break

                        scored = []
                        for p in candidates:
                            s_title = _simplify_ar(p.get('title_ar') or '')
                            s_desc = _simplify_ar(p.get('description_ar') or '')

                            # Count how many tokens appear in title or description (Arabic only)
                            match_count = 0
                            exact_count = 0
                            desc_bonus = 0
                            s_title_words = set(s_title.split())
                            s_desc_words = set(s_desc.split())
                            for tok in tokens:
                                if not tok:
                                    continue
                                if tok in s_title or tok in s_desc:
                                    match_count += 1
                                # exact whole-word check (Arabic simplified)
                                if tok in s_title_words or tok in s_desc_words:
                                    exact_count += 1
                                if tok in s_desc_words:
                                    desc_bonus += 1

                            # Compute detailed scoring breakdown
                            base = exact_count * 6 + match_count * 2 + desc_bonus * 2
                            phrase_bonus = 10 if all((tok in s_title or tok in s_desc) for tok in tokens if tok) else 0
                            symptom_boost = 0
                            domain_boost = 0

                            # Domain-specific heuristics map
                            symptom_map = {
                                'متقصف': {'tokens': ['متقصف', 'تقصف', 'تقصف الشعر'], 'cats': ['العناية بالشعر', 'شعر']},
                                'جاف': {'tokens': ['جاف', 'جافة'], 'cats': ['العناية بالبشرة', 'بشرة']},
                                'تساقط': {'tokens': ['تساقط', 'تساقط الشعر'], 'cats': ['العناية بالشعر', 'شعر']},
                                'حبوب': {'tokens': ['حبوب', 'حب الشباب'], 'cats': ['العناية بالبشرة', 'بشرة']},
                            }

                            # Symptom boost
                            for sym, info in symptom_map.items():
                                if any(sym_token in tok for tok in tokens for sym_token in info['tokens']):
                                    if any(sym_token in s_title or sym_token in s_desc for sym_token in info['tokens']):
                                        symptom_boost = 12
                                    else:
                                        cat = (p.get('category_ar') or p.get('category') or '').lower()
                                        attrs = p.get('attributes') or {}
                                        hair_type = [str(x).lower() for x in (attrs.get('hair_type') or [])]
                                        skin_type = [str(x).lower() for x in (attrs.get('skin_type') or [])]
                                        if any(c.lower() in cat for c in info['cats']) or any(any(sym_token in a for sym_token in info['tokens']) for a in hair_type + skin_type):
                                            symptom_boost = 6
                                    break

                            # Domain-level boost: if query suggests skin and product mentions 'بشر' or related, prefer it (and same for hair)
                            if domain == 'skin' and ('بشر' in s_title or 'بشر' in s_desc or 'بشرة' in (p.get('category_ar') or '') or 'بشرة' in (p.get('category') or '')):
                                domain_boost = 8
                            if domain == 'hair' and ('شعر' in s_title or 'شعر' in s_desc or 'شعر' in (p.get('category_ar') or '') or 'شعر' in (p.get('category') or '')):
                                domain_boost = 8

                            score = base + phrase_bonus + symptom_boost + domain_boost

                            # Protect against zero/negative scoring
                            if score <= 0:
                                score = match_count

                            # Append breakdown: (score, base, phrase_bonus, symptom_boost, domain_boost, exact_count, match_count, product)
                            scored.append((score, base, phrase_bonus, symptom_boost, domain_boost, exact_count, match_count, p))

                        # Sort by descending score, then exact_count, then match_count; return top results
                        # scored tuples: (score, base, phrase_bonus, symptom_boost, domain_boost, exact_count, match_count, product)
                        scored.sort(key=lambda x: (-x[0], -x[5], -x[6]))
                        results = [t[-1] for t in scored][:limit]

                        if results:
                            return results
                    except Exception:
                        # If pre-filter fails due to planning or server restrictions, fall back to normal search
                        pass

            # Use text search for better results in English queries; for Arabic queries search only Arabic fields
            if re.search(r'[\u0600-\u06FF]', query):
                filter_query["$or"] = [
                    {"title_ar": {"$regex": esc, "$options": "i"}},
                    {"description_ar": {"$regex": esc, "$options": "i"}}
                ]
            else:
                # Prefer text index for English/latin queries, but include regex fallback
                filter_query["$or"] = [
                    {"$text": {"$search": query}},
                    {"name": {"$regex": esc, "$options": "i"}},
                    {"title": {"$regex": esc, "$options": "i"}},
                    {"description": {"$regex": esc, "$options": "i"}},
                    {"title_ar": {"$regex": esc, "$options": "i"}},
                    {"description_ar": {"$regex": esc, "$options": "i"}}
                ]

        if category and category.strip():
            escaped_cat = re.escape(category.strip())
            # Partial, case-insensitive match
            filter_query["category"] = {"$regex": escaped_cat, "$options": "i"}
        
        if brand and brand.strip():
            escaped_brand = re.escape(brand.strip())
            # Partial, case-insensitive match
            filter_query["brand"] = {"$regex": escaped_brand, "$options": "i"}

        # If user asked for a product type (e.g., 'oil', 'shampoo'), match attributes.type or top-level 'type'
        if product_type and isinstance(product_type, str) and product_type.strip():
            esc_type = re.escape(product_type.strip())
            # Add as an AND condition: attributes.type OR type matches
            filter_query.setdefault('$and', [])
            filter_query['$and'].append({
                '$or': [
                    {"attributes.type": {"$regex": esc_type, "$options": "i"}},
                    {"type": {"$regex": esc_type, "$options": "i"}}
                ]
            })

        # Price range filters - if provided, we'll prefer to apply them client-side when DB min/max fields
        # are not reliably present (covers products storing prices in price_map/price dicts). We still keep
        # the DB-level filter when min_price/max_price fields exist to allow index use, but avoid rejecting
        # products that only store prices under 'price' or 'price_map'. We'll set a flag to indicate a range
        # filter is requested and apply post-filtering if necessary.
        price_filter_requested = False
        if min_price is not None or max_price is not None:
            price_filter_requested = True
            # Try to add DB filters if useful (this helps if documents have min_price/max_price indexed)
            try:
                if min_price is not None and max_price is not None:
                    # Prefer DB range match when documents have min_price/max_price, but allow docs that lack
                    # those fields to pass through (we'll do client-side filtering later). Use an $and with $or
                    # so we don't inadvertently exclude documents missing min/max fields.
                    filter_query.setdefault('$and', [])
                    filter_query['$and'].append({
                        '$or': [
                            {"max_price": {"$gte": float(min_price)}, "min_price": {"$lte": float(max_price)}},
                            {"min_price": {"$exists": False}},
                            {"max_price": {"$exists": False}}
                        ]
                    })
                elif min_price is not None:
                    filter_query.setdefault('$and', [])
                    filter_query['$and'].append({
                        '$or': [
                            {"max_price": {"$gte": float(min_price)}},
                            {"max_price": {"$exists": False}}
                        ]
                    })
                elif max_price is not None:
                    filter_query.setdefault('$and', [])
                    filter_query['$and'].append({
                        '$or': [
                            {"min_price": {"$lte": float(max_price)}},
                            {"min_price": {"$exists": False}}
                        ]
                    })
            except Exception:
                # If conversion fails, don't add DB filters; we'll filter in Python later
                pass
        
        # Execute query (some MongoDB server versions disallow $text inside $or with non-text clauses)
        try:
            cursor = self.products.find(filter_query, proj)
        except Exception as e:
            # Fall back to regex-only search if the server cannot plan $text under $or
            try:
                msg = str(e)
                if 'TEXT' in msg or 'Failed to produce a solution for TEXT under OR' in msg or getattr(e, 'code', None) == 291:
                    # Remove any $text clause from $or
                    if '$or' in filter_query:
                        new_or = [cl for cl in filter_query['$or'] if not ('$text' in cl)]
                        if new_or:
                            filter_query['$or'] = new_or
                        else:
                            filter_query.pop('$or', None)
                    cursor = self.products.find(filter_query, {
                        "_id": 0,
                        "product_id": 1,
                        "name": 1,
                        "title": 1,
                        "title_ar": 1,
                        "description": 1,
                        "price": 1,
                        "min_price": 1,
                        "max_price": 1,
                        "price_map": 1,
                        "original_price": 1,
                        "currency": 1,
                        "category": 1,
                        "brand": 1,
                        "in_stock": 1,
                        "stock_quantity": 1,
                        "attributes": 1,
                        "images": 1,
                        "image_url": 1,
                        "rating": 1,
                        "review_count": 1
                    })
                else:
                    raise
            except Exception:
                # Re-raise original error if fallback fails
                raise e

        # Apply sorting if requested (support sorting by price, rating, etc.)
        if sort_by:
            order = ASCENDING if sort_order == 1 else DESCENDING
            cursor = cursor.sort(sort_by, order)

        cursor = cursor.limit(limit)
        
        # Attempt to materialize cursor; some servers raise on execution planning when $text is used inside $or
        try:
            products = list(cursor)
        except Exception as e:
            # Retry without $text clause in the $or if present
            try:
                msg = str(e)
                if 'TEXT' in msg or 'Failed to produce a solution for TEXT under OR' in msg or getattr(e, 'code', None) == 291:
                    if '$or' in filter_query:
                        new_or = [cl for cl in filter_query['$or'] if not ('$text' in cl)]
                        if new_or:
                            filter_query['$or'] = new_or
                        else:
                            filter_query.pop('$or', None)
                    cursor = self.products.find(filter_query, {
                        "_id": 0,
                        "product_id": 1,
                        "name": 1,
                        "title": 1,
                        "title_ar": 1,
                        "description": 1,
                        "price": 1,
                        "min_price": 1,
                        "max_price": 1,
                        "price_map": 1,
                        "original_price": 1,
                        "currency": 1,
                        "category": 1,
                        "brand": 1,
                        "in_stock": 1,
                        "stock_quantity": 1,
                        "attributes": 1,
                        "images": 1,
                        "image_url": 1,
                        "rating": 1,
                        "review_count": 1
                    })
                    cursor = cursor.limit(limit)
                    products = list(cursor)
                else:
                    raise
            except Exception:
                raise e

        # If a price range was requested, perform client-side filtering to support products storing per-size prices
        if price_filter_requested:
            try:
                # Start with the already fetched products; to improve hit-rate, fetch a larger candidate set if needed
                candidates = products
                if len(candidates) < limit:
                    more_cursor = self.products.find(filter_query, {
                        "_id": 0,
                        "product_id": 1,
                        "name": 1,
                        "title": 1,
                        "title_ar": 1,
                        "description": 1,
                        "price": 1,
                        "min_price": 1,
                        "max_price": 1,
                        "price_map": 1,
                        "original_price": 1,
                        "currency": 1,
                        "category": 1,
                        "brand": 1,
                        "in_stock": 1,
                        "stock_quantity": 1,
                        "attributes": 1,
                        "images": 1,
                        "image_url": 1,
                        "rating": 1,
                        "review_count": 1
                    }).limit(max(200, limit*5))
                    try:
                        candidates = list(more_cursor)
                    except Exception:
                        pass

                def _prod_min_price(prod):
                    # Determine the minimum numeric price for a product from available fields
                    mp = prod.get('min_price')
                    if mp is not None:
                        try:
                            return float(mp)
                        except Exception:
                            pass
                    pm_raw = prod.get('price_map') or prod.get('price') or prod.get('pricing') or {}
                    vals = []
                    if isinstance(pm_raw, dict):
                        for v in pm_raw.values():
                            if isinstance(v, dict):
                                vv = v.get('price') or v.get('value') or v.get('amount')
                            else:
                                vv = v
                            try:
                                vals.append(float(vv))
                            except Exception:
                                continue
                    elif isinstance(pm_raw, list):
                        for itm in pm_raw:
                            if isinstance(itm, dict):
                                vv = itm.get('price') or itm.get('value') or itm.get('amount')
                                try:
                                    vals.append(float(vv))
                                except Exception:
                                    continue
                    if vals:
                        return min(vals)
                    # fallback to single numeric price
                    p = prod.get('price')
                    try:
                        return float(p)
                    except Exception:
                        return None

                filtered = []
                for p in candidates:
                    pmin = _prod_min_price(p)
                    if pmin is None:
                        continue
                    ok = True
                    if min_price is not None and pmin < float(min_price):
                        ok = False
                    if max_price is not None and pmin > float(max_price):
                        ok = False
                    if ok:
                        filtered.append((pmin, p))

                filtered.sort(key=lambda x: x[0])
                products = [p for _, p in filtered][:limit]
            except Exception as e:
                logger.warning(f"Client-side price filtering failed: {e}")

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
                        cursor2 = self.products.find({"name": {"$in": matched_originals}}, proj)
                        if sort_by == 'price':
                            order = ASCENDING if sort_order == 1 else DESCENDING
                            cursor2 = cursor2.sort('price', order)
                        cursor2 = cursor2.limit(limit)
                        products = list(cursor2)
            except Exception:
                # If fuzzy fallback fails, return the empty result we already have
                pass

        # If a product_type filter was supplied, perform local filtering to support both Atlas and in-memory DBs
        if product_type and isinstance(product_type, str) and product_type.strip():
            lp = product_type.strip().lower()
            def _matches_type(p):
                attrs = p.get('attributes') or {}
                t = attrs.get('type') or p.get('type') or ''
                return lp in str(t).lower()
            products = [p for p in products if _matches_type(p)]

        return products
    
    def get_product_by_name(self, product_name: str) -> Optional[Dict]:
        """Get product by name or title (fuzzy match).

        Searches both English and Arabic title fields and description fields to improve
        matching for multilingual content.
        """
        if not product_name or not str(product_name).strip():
            return None
        try:
            esc = re.escape(str(product_name))
            q = {
                "$or": [
                    {"name": {"$regex": esc, "$options": "i"}},
                    {"title": {"$regex": esc, "$options": "i"}},
                    {"title_ar": {"$regex": esc, "$options": "i"}},
                    {"description": {"$regex": esc, "$options": "i"}},
                    {"description_ar": {"$regex": esc, "$options": "i"}}
                ]
            }
            product = self.products.find_one(q)
            return product
        except Exception:
            # Fallback to simple find_one with the raw product_name
            return self.products.find_one({"$or": [{"name": product_name}, {"title": product_name}]})
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        """Get product by product_id or _id (accepts string ObjectId)"""
        q_or = [{"product_id": product_id}]
        try:
            q_or.append({"_id": ObjectId(product_id)})
        except Exception:
            q_or.append({"_id": product_id})
        product = self.products.find_one({"$or": q_or})
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

    # --- Shopping cart helpers ---
    def create_cart(self, customer_name: str, phone: str) -> Dict[str, Any]:
        """Create a new shopping cart for a customer"""
        import uuid
        cart_id = str(uuid.uuid4())
        cart = {
            "cart_id": cart_id,
            "customer": {"name": customer_name, "phone": phone},
            "items": [],
            "currency": "SAR",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self.carts.insert_one(cart)
        return self.get_cart_by_id(cart_id)

    def get_cart_by_id(self, cart_id: str) -> Optional[Dict[str, Any]]:
        return self.carts.find_one({"cart_id": cart_id}, {"_id": 0})

    def get_cart_by_customer(self, name: str, phone: str) -> Optional[Dict[str, Any]]:
        # case-insensitive name match and phone match
        return self.carts.find_one({
            "customer.name": {"$regex": f"^{re.escape((name or '').strip())}$", "$options": "i"},
            "customer.phone": {"$regex": re.escape((phone or '').strip())}
        }, {"_id": 0})

    def get_or_create_cart_by_customer(self, name: str, phone: str) -> Dict[str, Any]:
        cart = self.get_cart_by_customer(name, phone)
        if cart:
            return cart
        return self.create_cart(name, phone)

    def add_item_to_cart(self, cart_id: str, product_id: str, quantity: int = 1, size: str = None, unit_price: float = None) -> Dict[str, Any]:
        """Add a product to the cart, include item snapshot for price stability.
        Supports per-size pricing by passing `size` and/or `unit_price`.
        """
        product = self.get_product_by_id(product_id)
        if not product:
            raise Exception("Product not found")

        # Determine price: preference order -> explicit unit_price, price_map[size], min_price, price
        price_val = None
        if unit_price is not None:
            try:
                price_val = float(unit_price)
            except Exception:
                price_val = unit_price  # fall back to original for later coercion
        if price_val is None and size and isinstance(product.get('price_map'), dict):
            try:
                pm = product.get('price_map')
                # price_map keys might be strings like '50ml'
                if size in pm:
                    price_val = pm[size]
            except Exception:
                price_val = None
        if price_val is None:
            price_val = product.get('min_price') or product.get('price') or 0

        # Coerce price_val into a numeric float when possible
        try:
            if isinstance(price_val, dict):
                # try common keys
                for k in ('price', 'amount', 'value'):
                    if k in price_val:
                        try:
                            price_val = float(price_val[k])
                            break
                        except Exception:
                            continue
                else:
                    # try first numeric-like inner value
                    found = False
                    for v in price_val.values():
                        try:
                            price_val = float(v)
                            found = True
                            break
                        except Exception:
                            continue
                    if not found:
                        price_val = 0.0
            price_val = float(price_val)
        except Exception:
            price_val = 0.0

        item = {
            "product_id": product.get('product_id'),
            "name": product.get('name') or product.get('title'),
            "price": price_val,
            "currency": product.get('currency') or 'SAR',
            "quantity": int(quantity),
            "size": size,
            "added_at": datetime.now()
        }
        self.carts.update_one({"cart_id": cart_id}, {"$push": {"items": item}, "$set": {"updated_at": datetime.now()}})
        cart = self.get_cart_by_id(cart_id)
        # compute subtotal safely
        subtotal = 0.0
        for it in cart.get('items', []):
            p = it.get('price') or 0
            try:
                pnum = float(p)
            except Exception:
                pnum = 0.0
            q = it.get('quantity') or 1
            try:
                qnum = int(q)
            except Exception:
                qnum = 1
            subtotal += pnum * qnum
        cart['subtotal'] = subtotal
        return cart

    def get_cart_summary(self, cart_id: str) -> Optional[Dict[str, Any]]:
        cart = self.get_cart_by_id(cart_id)
        if not cart:
            return None
        subtotal = sum([(it.get('price') or 0) * (it.get('quantity') or 1) for it in cart.get('items', [])])
        cart['subtotal'] = subtotal
        cart['items_count'] = sum([it.get('quantity') or 1 for it in cart.get('items', [])])
        return cart

    def checkout_cart(self, cart_id: str) -> Optional[Dict[str, Any]]:
        """Mark a cart as checked out and clear its items (simple simulation of checkout)."""
        try:
            res = self.carts.update_one({"cart_id": cart_id}, {"$set": {"checked_out": True, "checked_out_at": datetime.now(), "items": []}})
            if res.modified_count:
                return self.get_cart_by_id(cart_id)
        except Exception as e:
            logger.warning(f"Checkout failed for cart {cart_id}: {e}")
        return None

    def delete_cart(self, cart_id: str) -> bool:
        """Delete a cart entirely (used when user ends session and wants to clear cart)."""
        try:
            res = self.carts.delete_one({"cart_id": cart_id})
            return res.deleted_count > 0
        except Exception as e:
            logger.warning(f"Delete cart failed for {cart_id}: {e}")
            return False

    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Create singleton instance
mongo_service = MongoDBService()