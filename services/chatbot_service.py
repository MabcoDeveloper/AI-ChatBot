import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Access product data for brand/category extraction
from services.mongo_service import mongo_service
import uuid
from models.memory_manager import memory_manager as conv_memory
from models.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

class ArabicNormalizer:
    """Arabic text normalizer with Syrian dialect support"""
    
    def __init__(self):
        self.dialect_mappings = {
            "وين": "أين",
            "شو": "ما",
            "بدي": "أريد",
            "عندكم": "لديكم",
            "بكم": "بكم",
            "شغلات": "أشياء",
            "متل": "مثل",
            "هاي": "هذه",
            "كتير": "كثير",
            "منيح": "جيد",
            "طيب": "حسنا"
        }
    
    def normalize(self, text: str) -> str:
        """Normalize Arabic text (lowercase, dialect mappings, collapse accidental repeats)"""
        if not text:
            return ""

        normalized = text.strip()
        # Lowercase for consistent matching
        normalized = normalized.lower()

        # Replace dialect words using word boundaries to avoid accidental substrings
        for dialect, standard in self.dialect_mappings.items():
            pattern = r'\b' + re.escape(dialect) + r'\b'
            normalized = re.sub(pattern, standard, normalized, flags=re.IGNORECASE)

        # Collapse accidental repeated Arabic letters (e.g., 'المنتتجات' -> 'المنتجات')
        normalized = re.sub(r'([\u0621-\u064A])\1+', r'\1', normalized)

        # Remove punctuation and extra whitespace
        normalized = re.sub(r'[^\u0621-\u064A0-9\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

class IntentDetector:
    """Simple intent detector for Arabic with safer keyword matching"""
    
    def __init__(self):
        self.search_keywords = ["ابحث", "عندكم", "لديكم", "وين", "أين", "بدي", "أريد", "شو في", "ما في", "منتج", "منتجات"]
        self.price_keywords = ["سعر", "كم", "بكم", "السعر", "تكلفة"]
        self.offer_keywords = ["عروض", "خصومات", "تخفيضات", "عرض", "تنزيلات"]
        self.help_keywords = ["مساعدة", "مساعدة عامة", "شو بتقدر", "كيف", "طريقة"]
        self.greeting_keywords = ["مرحبا", "السلام عليكم", "اهلا", "صباح الخير", "مساء الخير"]
        # Closing / thanks phrases
        self.thanks_keywords = ["شكرا", "شكراً", "شكرا جزيلا", "شكرا لك", "ممنون", "مشكور", "تسلم", "جزاك", "مع السلامة", "وداعا", "bye", "thanks", "thank you"]
        # Buy / add to cart keywords
        self.buy_keywords = ["اشتري", "شراء", "اضف للسلة", "اضف الى السلة", "اضف للعربة", "أضيف", "اضافة للسلة", "اشتريه", "أريد شراء"]
        # End session keywords
        self.end_keywords = ["انهاء", "انهي", "خلاص", "انتهى", "خروج", "انهاء المحادثة", "انهاء الجلسة"]
    
    def _contains_keyword(self, text: str, kw: str) -> bool:
        """Match keyword only when not embedded inside other Arabic letters or numbers.
        This avoids false positives such as matching 'كم' inside 'لديكم'.
        Uses lookarounds so punctuation like '؟' is treated as a valid boundary."""
        # Ensure the keyword is not part of a larger Arabic word or number by asserting
        # the surrounding chars are not Arabic letters or digits.
        # Use a tighter Arabic letter range (0621-064A) and include Arabic-Indic digits (0660-0669)
        pattern = rf'(?<![\u0621-\u064A\u0660-\u06690-9]){re.escape(kw)}(?![\u0621-\u064A\u0660-\u06690-9])'
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    def detect(self, text: str) -> Dict[str, Any]:
        """Detect intent from text using token-aware matching"""
        text_lower = text.lower()
        
        # Check for greetings first
        if any(self._contains_keyword(text_lower, greet) for greet in self.greeting_keywords):
            return {"intent": "greeting", "confidence": 0.95}
        
        # Check for closing / thanks (handle before search to avoid false fallback)
        if any(self._contains_keyword(text_lower, kw) for kw in self.thanks_keywords):
            return {"intent": "closing", "confidence": 0.98}
        
        # Detect numeric price-range patterns (e.g., 'بين 10 و 100', 'من 10 الى 50', 'اقل من 50')
        if re.search(r'(?:بين|من)\s*[0-9٠-٩]+\s*(?:و|الى|إلى)\s*[0-9٠-٩]+', text_lower) or 'اقل من' in text_lower or 'اكبر من' in text_lower:
            return {"intent": "price", "confidence": 0.92}

        # Check for best product requests (e.g., 'أفضل شامبو', 'ما هو أفضل منتج')
        best_kw = ['أفضل', 'افضل', 'الأفضل']
        if any(self._contains_keyword(text_lower, kw) for kw in best_kw) or 'أفضل' in text_lower or 'افضل' in text_lower:
            return {"intent": "best", "confidence": 0.92}

        # Check for price intent (words like 'ارخص', 'اغلى', 'سعر') and ensure price intent takes precedence
        price_indicators = ['سعر', 'ارخص', 'الأرخص', 'اغلى', 'اعلى', 'الأغلى']
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.price_keywords) or any(ind in text_lower for ind in price_indicators):
            return {"intent": "price", "confidence": 0.90}

        # Check for buy / add-to-cart intent (relaxed check for common keywords)
        if any(substr in text_lower for substr in ['شراء', 'اشتري', 'سلة', 'اضف']):
            return {"intent": "buy", "confidence": 0.95}
        if any(self._contains_keyword(text_lower, kw) or kw in text_lower for kw in getattr(self, 'buy_keywords', [])):
            return {"intent": "buy", "confidence": 0.92}

        # Check for explicit end-session requests
        if any(kw in text_lower for kw in getattr(self, 'end_keywords', [])):
            return {"intent": "end_session", "confidence": 0.98}

        # Check for offers intent (higher precedence than generic search)
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.offer_keywords):
            return {"intent": "offers", "confidence": 0.95}
        # Check for search intent
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.search_keywords):
            return {"intent": "search", "confidence": 0.90}        
        
        # Check for help intent
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.help_keywords):
            return {"intent": "help", "confidence": 0.85}
        
        # Default to fallback
        return {"intent": "fallback", "confidence": 0.50}

class ChatbotService:
    """Main chatbot service"""
    
    def __init__(self):
        self.normalizer = ArabicNormalizer()
        self.intent_detector = IntentDetector()
        self.memory = {}  # Simple in-memory conversation storage
        # Keep the last search summaries per user to support follow-up requests
        self.search_context: Dict[str, List[Dict[str, Any]]] = {}
        # Per-user transient state (pending buy info, etc.) - in-memory fallback
        self.user_state: Dict[str, Dict[str, Any]] = {}
        # Conversation memory (Redis) instance
        self.conv_memory = conv_memory
        # Intent model placeholder (loadable)
        self.intent_model = None
        try:
            self.intent_model = IntentClassifier()
            self.intent_model.load('models/intent_model.joblib')
            logger.info('✅ Loaded intent model from models/intent_model.joblib')
        except Exception:
            logger.info('No intent model found, using rule-based detector')
        # Aliases for categories and brands (map common user words to canonical names)
        # Values can be lists; we try them in order when searching
        # Map common Arabic user terms to either English categories (as stored in assets)
        # or to product types (attributes.type) so searches like 'زيوت' or 'شامبو' still work
        self.category_aliases: Dict[str, List[str]] = {
            'زيوت': ['oil', 'Hair Care'],
            'زيت': ['oil', 'Hair Care'],
            'شامبو': ['shampoo', 'Hair Care'],
            'بلسم': ['conditioner', 'Hair Care'],
            'كريم': ['العناية بالبشرة', 'Skin Care', 'Body Care'],
            'مرطب': ['العناية بالبشرة', 'Skin Care'],
            'لوشن': ['lotion', 'Body Care'],
            'عطر': ['العطور', 'Fragrances'],
            'عطور': ['العطور', 'Fragrances'],
            'مستحضرات': ['Makeup', 'مكياج'],
            'مكياج': ['Makeup']
        }
        self.brand_aliases: Dict[str, List[str]] = {
            'نيفيا': ['نيڤيا'],
            'ماك': ['ماك'],
            'ذا أورديناري': ['ذا أورديناري']
        }
        
    # Helper wrappers to access conversation state and simple training logs from other methods
    def _get_state(self, user_id: str) -> Dict[str, Any]:
        try:
            st = self.conv_memory.get_user_state(user_id)
            return st or {}
        except Exception:
            return self.user_state.get(user_id, {})

    def _set_state(self, user_id: str, st: Dict[str, Any]):
        try:
            self.conv_memory.set_user_state(user_id, st)
        except Exception:
            self.user_state[user_id] = st

    def _clear_state(self, user_id: str):
        try:
            # best-effort: if the conv_memory supports clearing, use it
            if hasattr(self.conv_memory, 'clear_user_state'):
                self.conv_memory.clear_user_state(user_id)
            else:
                self.conv_memory.set_user_state(user_id, {})
        except Exception:
            if user_id in self.user_state:
                del self.user_state[user_id]

    def _log_bot_turn(self, user_id: str, bot_message: str, intent_label: str = None, metadata: Dict = None):
        try:
            turn = {
                'role': 'bot',
                'message': bot_message,
                'intent': intent_label,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            # Append to in-memory conversation log; training storage is best-effort
            self.memory.setdefault(user_id, []).append(turn)
        except Exception:
            pass

    def _log_user_turn(self, user_id: str, message: str, normalized_msg: str, detected_intent: str = None, metadata: Dict = None):
        try:
            turn = {
                'role': 'user',
                'message': message,
                'normalized': normalized_msg,
                'intent': detected_intent,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            self.memory.setdefault(user_id, []).append(turn)
        except Exception:
            pass

    def _get_price_info(self, prod: Dict[str, Any]) -> Dict[str, Any]:

        # Accept price map from several common fields: 'price_map', 'price' (sometimes used as dict), or 'pricing'
        pm_raw = (prod or {}).get('price_map') or (prod or {}).get('price') or (prod or {}).get('pricing') or {}
        price_map: Dict[str, float] = {}
        sizes: List[str] = []
        # Normalize dict-like price maps
        if isinstance(pm_raw, dict):
            for k, v in pm_raw.items():
                try:
                    price_map[k] = float(v)
                except Exception:
                    price_map[k] = v
            sizes = list(price_map.keys())
        # Support list-of-dicts price_map formats
        elif isinstance(pm_raw, list):
            for itm in pm_raw:
                if isinstance(itm, dict):
                    s = itm.get('size') or itm.get('label') or itm.get('name')
                    p = itm.get('price') or itm.get('value')
                    if s:
                        try:
                            price_map[s] = float(p)
                        except Exception:
                            price_map[s] = p
                        sizes.append(s)
        # Compute minimum numeric price when available
        min_price = None
        try:
            numeric_vals = [v for v in price_map.values() if isinstance(v, (int, float))]
            if numeric_vals:
                min_price = min(numeric_vals)
        except Exception:
            min_price = None
        if min_price is None:
            mp = prod.get('min_price') if prod.get('min_price') is not None else prod.get('price')
            try:
                min_price = float(mp) if mp is not None else None
            except Exception:
                min_price = mp
        return {'min_price': min_price, 'price_map': price_map, 'sizes': sizes}

    def _is_in_stock(self, prod: Dict[str, Any], size: Optional[str] = None) -> bool:
        """Return True if the product (or a specific size/variant) appears to be in stock.

        Rules implemented:
        - If a size is provided and that variant contains explicit 'stock' info, use it.
        - If the variant has no explicit 'stock', prefer product-level `in_stock`/`inStock` when present; otherwise assume available when the product defines sizes/prices (per your requirement).
        - When checking overall product availability (no size specified): prefer explicit product flags, then numeric quantities; if variants exist and none provide explicit stock info, assume available; if variants provide stock info use that to determine availability.
        - Fall back to free-text tokens; default False only if no signals indicate availability.
        """
        if prod is None:
            return False

        # Normalize variants/prices
        pinfo = self._get_price_info(prod)
        pm = pinfo.get('price_map') or {}

        # Size-specific handling
        if size and isinstance(pm, dict) and size in pm:
            v = pm.get(size)
            # Explicit per-variant stock key should be honored when present
            if isinstance(v, dict) and 'stock' in v:
                try:
                    return int(v.get('stock') or 0) > 0
                except Exception:
                    pass
            # No explicit per-variant stock -> prefer product-level boolean flags if present
            if prod.get('in_stock') is not None:
                return bool(prod.get('in_stock'))
            if prod.get('inStock') is not None:
                return bool(prod.get('inStock'))
            # If no product-level flags and variant has no explicit stock, assume available
            return True

        # No size specified: check variants for explicit stock info
        if isinstance(pm, dict) and pm:
            any_variant_has_stock_info = False
            for v in pm.values():
                if isinstance(v, dict) and 'stock' in v:
                    any_variant_has_stock_info = True
                    try:
                        if int(v.get('stock') or 0) > 0:
                            return True
                    except Exception:
                        continue
            # If variants had explicit stock info but none > 0 -> not available
            if any_variant_has_stock_info:
                return False
            # If variants exist but no explicit stock info, assume available
            return True

        # Product-level boolean flags
        if prod.get('in_stock') is not None:
            return bool(prod.get('in_stock'))
        if prod.get('inStock') is not None:
            return bool(prod.get('inStock'))

        # Numeric quantity
        qty = prod.get('stock_quantity') if prod.get('stock_quantity') is not None else prod.get('quantity')
        try:
            if qty is not None:
                return int(qty) > 0
        except Exception:
            pass

        # Free-text tokens
        avail = str(prod.get('availability') or '').lower()
        if any(token in avail for token in ('متوفر', 'available', 'in stock', 'متاح')):
            return True

        # Default: not available
        return False

    def process_message(self, user_id: str, message: str, **kwargs):
        """
        Process user message - FIXED: removed session_data parameter
        Accepts **kwargs to handle extra parameters from UI
        """
        logger.info(f"Processing message from user {user_id}: {message}")
        
        # Normalize the message
        normalized = self.normalizer.normalize(message)
        
        # Detect intent - prefer ML model if available
        intent = None
        intent_confidence = 0.0
        intent_result = None
        try:
            if self.intent_model and getattr(self.intent_model, 'pipeline', None):
                pred = self.intent_model.predict(normalized)
                probs = self.intent_model.predict_proba(normalized)
                intent = pred
                intent_confidence = float(max(probs.values())) if isinstance(probs, dict) and probs else 0.0
                intent_result = {'intent': intent, 'confidence': intent_confidence, 'source': 'ml'}
            else:
                intent_result = self.intent_detector.detect(normalized)
                intent = intent_result['intent']
                intent_confidence = intent_result.get('confidence', 0.0)
        except Exception:
            intent_result = self.intent_detector.detect(normalized)
            intent = intent_result['intent']
            intent_confidence = intent_result.get('confidence', 0.0)
        
        # Store in memory
        if user_id not in self.memory:
            self.memory[user_id] = []
        self.memory[user_id].append({
            "role": "user",
            "message": message,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })

        # Helper: short affirmative detection (e.g., 'نعم', 'أيوه', 'yes')
        def _is_affirmative(s: str) -> bool:
            if not s:
                return False
            s = s.strip().lower()
            aff = ['نعم', 'ايوه', 'أيوه', 'ايه', 'نعمً', 'نعمًا', 'نعم', 'نعم', 'نعم', 'yes', 'y', 'أريد', 'عايز', 'أيوه']
            return any(w == s or s == w for w in aff)

        # State helpers to use persistent conversation memory when available
        def _get_state(u_id: str) -> Dict[str, Any]:
            try:
                st = self.conv_memory.get_user_state(u_id) or {}
                # ensure it's a plain dict
                return dict(st)
            except Exception:
                return self.user_state.get(u_id, {})

        def _set_state(u_id: str, st: Dict[str, Any]):
            try:
                self.conv_memory.set_user_state(u_id, st)
            except Exception:
                self.user_state[u_id] = st

        def _clear_state(u_id: str):
            try:
                self.conv_memory.clear_user_state(u_id)
            except Exception:
                self.user_state.pop(u_id, None)

        

        # Training/session logging helpers
        def _start_training_session(u_id: str) -> str:
            st = _get_state(u_id)
            session_id = st.get('training_session_id') or str(uuid.uuid4())
            # create session doc
            doc = {
                'session_id': session_id,
                'user_id': u_id,
                'created_at': datetime.utcnow(),
                'turns': [],
                'outcome': None
            }
            try:
                mongo_service.insert_training_session(doc)
                st['training_session_id'] = session_id
                _set_state(u_id, st)
            except Exception:
                pass
            return session_id

        def _log_user_turn(u_id: str, message: str, normalized_msg: str, detected_intent: str = None, metadata: Dict = None):
            st = _get_state(u_id)
            sid = st.get('training_session_id') or _start_training_session(u_id)
            turn = {
                'role': 'user',
                'message': message,
                'normalized': normalized_msg,
                'intent': detected_intent,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            try:
                mongo_service.log_training_turn(sid, turn)
            except Exception:
                pass

        def _log_bot_turn(u_id: str, bot_message: str, intent_label: str = None, metadata: Dict = None):
            st = _get_state(u_id)
            sid = st.get('training_session_id') or _start_training_session(u_id)
            turn = {
                'role': 'bot',
                'message': bot_message,
                'intent': intent_label,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            try:
                mongo_service.log_training_turn(sid, turn)
            except Exception:
                pass

        def _finalize_training_session(u_id: str, outcome_label: str, summary: Dict = None):
            st = _get_state(u_id)
            sid = st.get('training_session_id')
            if not sid:
                return False
            try:
                mongo_service.finalize_training_session(sid, outcome_label, summary)
                # clear session id from state
                st.pop('training_session_id', None)
                _set_state(u_id, st)
                return True
            except Exception:
                return False

        # If we have a pending buy awaiting customer info or confirmation, handle each step
        state = self._get_state(user_id)
        pending_buy = state.get('pending_buy') if state else None

        # Log user turn for training purposes
        try:
            self._log_user_turn(user_id, message, normalized, intent)
        except Exception:
            pass

        # QUICK-FIX: If user simply replies 'yes' while we have a pending buy or a selected product,
        # handle as a confirmation regardless of intermediate awaiting flags. This prevents cases
        # where number-based selection or search-context branches accidentally leave the state
        # in 'size_selection' and a later 'نعم' returns nothing.
        if _is_affirmative(normalized) and (pending_buy or state.get('selected_product_id')):
            try:
                prod = pending_buy.get('product') if pending_buy else state.get('last_viewed_product')
                chosen_size = pending_buy.get('size') if pending_buy else None
                # Clear pending buy since the user confirmed
                state.pop('pending_buy', None)
                state.pop('awaiting_size_selection', None)
                _set_state(user_id, state)

                response_text = "تم بدء طلب إضافة المنتج إلى سلة التسوق بنجاح."
                product_title = ( prod.get('title') or prod.get('name')) if prod else None
                data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": chosen_size, "purchase_intent": True, "status": "started", "ask_confirm": False}
                intent = 'buy'
                self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                try:
                    self._log_bot_turn(user_id, response_text, 'buy', {'product_title': product_title, 'selected_size': chosen_size})
                except Exception:
                    pass
                return self._format_response(user_id, message, normalized, intent, 0.95, response_text, data=data, suggestions=[])
            except Exception as e:
                logger.warning(f"Quick confirm handling failed: {e}")
                return {"user_id": user_id, "original_message": message, "normalized_message": normalized, "intent": 'buy', "intent_confidence": 0.0, "response": "عذراً، حدث خطأ عند معالجة التأكيد.", "data": None, "suggestions": []}

        # If the user provides name+phone while we are awaiting a SIZE selection for a pending buy,
        # prompt for the size first instead of proceeding to add to cart.
        cust_direct = self._parse_customer_info(normalized)
        if cust_direct.get('phone') and pending_buy and pending_buy.get('awaiting') == 'size_selection':
            try:
                prod = pending_buy.get('product') or state.get('last_viewed_product')
                pm = (prod or {}).get('price_map') or {}
                if isinstance(pm, dict) and len(pm) > 1:
                    sizes = list(pm.keys())
                    choices = []
                    for i, s in enumerate(sizes):
                        val = pm.get(s)
                        # support dict variant entries like {'price': 15}
                        if isinstance(val, dict):
                            pv = val.get('price') or val.get('value') or val.get('amount')
                        else:
                            pv = val
                        price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                        avail = "متوفر" if self._is_in_stock(prod, size=s) else "غير متوفر"
                        choices.append(f"{i+1}. {s} - {price_str} {prod.get('currency','USD')} - {avail}")
                    state['awaiting_size_selection'] = {'product': prod, 'sizes': sizes}
                    _set_state(user_id, state)
                    response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                    try:
                        _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                    except Exception:
                        pass
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'clarify',
                        "intent_confidence": 0.80,
                        "response": response_text,
                        "data": {"sizes": sizes},
                        "suggestions": [],
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": 'clarify'
                        },
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception:
                pass

        # Allow users to provide name+phone directly after viewing product details
        # even if they didn't say 'نعم' explicitly to start the buy flow. Also handle 'اشتري' targeted at last_viewed_product
        if (not pending_buy) and state and state.get('last_viewed_product'):
            # If the user explicitly expressed a buy intent while a product is being viewed, start buy flow for that product
            if intent == 'buy' or re.search(r'\bاشتري\b', normalized.lower()):
                prod = state.get('last_viewed_product')
                # If the stored last_viewed_product lacks identifiers, attempt to fetch the full DB document
                if not prod.get('product_id') and not prod.get('_id'):
                    try:
                        orig = mongo_service.get_product_by_name(prod.get('title') or prod.get('title_ar') or prod.get('name') or '')
                        if orig:
                            prod = orig
                            state['last_viewed_product'] = prod
                    except Exception:
                        pass
                pinfo = self._get_price_info(prod)
                sizes = pinfo.get('sizes') or []
                if sizes and len(sizes) > 1:
                    choices = []
                    for i, s in enumerate(sizes):
                        val = pinfo['price_map'].get(s)
                        if isinstance(val, dict):
                            pv = val.get('price') or val.get('value') or val.get('amount')
                        else:
                            pv = val
                        price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                        avail = "متوفر" if self._is_in_stock(prod, size=s) else "غير متوفر"
                        choices.append(f"{i+1}. {s} - {price_str} {prod.get('currency','USD')} - {avail}")
                    state['awaiting_size_selection'] = {'product': prod, 'sizes': sizes}
                    state['pending_buy'] = {'product': prod, 'awaiting': 'size_selection'}
                    state['selected_product_id'] = self._normalize_pid(prod.get('product_id') or prod.get('_id'))
                    _set_state(user_id, state)
                    response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                    try:
                        _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                    except Exception:
                        pass
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'clarify',
                        "intent_confidence": 0.85,
                        "response": response_text,
                        "data": {"sizes": sizes},
                        "suggestions": ["1", "2"],
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "recent_questions": _get_state(user_id).get('recent_questions', [])
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # single-size product: directly add to cart as guest (no name/phone required)
                    pinfo = self._get_price_info(prod)
                    pm = pinfo.get('price_map') or {}
                    chosen_size = None
                    chosen_price = None
                    if isinstance(pm, dict) and len(pm) == 1:
                        chosen_size = list(pm.keys())[0]
                        try:
                            chosen_price = float(list(pm.values())[0])
                        except Exception:
                            chosen_price = list(pm.values())[0]
                    else:
                        chosen_price = pinfo.get('min_price') if pinfo.get('min_price') is not None else pinfo.get('price')

                    pid = self._normalize_pid(prod.get('product_id') or prod.get('_id'))

                    # Do NOT create cart or add to cart here. Record the pending buy and ask the user to confirm the order (no name/phone required)
                    state['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': 1, 'size': chosen_size, 'unit_price': chosen_price}
                    state['selected_product_id'] = pid
                    _set_state(user_id, state)

                    # Save intent in memory and inform user we started the add-to-cart flow and ask to confirm
                    self.memory.setdefault(user_id, []).append({
                        "role": "bot",
                        "message": f"تم بدء إضافة {prod.get('title') or prod.get('name')} إلى سلة التسوق. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء.",
                        "intent": 'buy',
                        "timestamp": datetime.now().isoformat()
                    })

                    response_text = f"تم بدء إضافة {prod.get('title') or prod.get('name')} إلى سلة التسوق بنجاح. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                    data = {
                        "action": "start_add_to_cart",
                        "product_id": pid,
                        "selected_size": chosen_size,
                        "purchase_intent": True,
                        "status": "started",
                        "ask_confirm": True
                    }
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "recent_questions": _get_state(user_id).get('recent_questions', [])
                        },
                        "timestamp": datetime.now().isoformat()
                    }

            cust_direct = self._parse_customer_info(normalized)
            if cust_direct.get('phone'):
                try:
                    prod = state.get('last_viewed_product')
                    pinfo = self._get_price_info(prod)
                    pm = pinfo.get('price_map') or {}
                    # If product has multiple sizes, require size selection first
                    if isinstance(pm, dict) and len(pm) > 1:
                        sizes = list(pm.keys())
                        choices = []
                        for i, s in enumerate(sizes):
                            val = pm.get(s)
                            if isinstance(val, dict):
                                pv = val.get('price') or val.get('value') or val.get('amount')
                            else:
                                pv = val
                            price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                            choices.append(f"{i+1}. {s} - {price_str} {prod.get('currency','USD')}")
                        state['awaiting_size_selection'] = {'product': prod, 'sizes': sizes}
                        _set_state(user_id, state)
                        response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                        try:
                            _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                        except Exception:
                            pass
                        return {
                            "user_id": user_id,
                            "original_message": message,
                            "normalized_message": normalized,
                            "intent": 'clarify',
                            "intent_confidence": 0.80,
                            "response": response_text,
                            "data": {"sizes": sizes},
                            "suggestions": [],
                            "context_summary": {
                                "turns_count": len(self.memory.get(user_id, [])),
                                "last_activity": datetime.now().isoformat(),
                                "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                                "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                                "last_intent": 'clarify'
                            },
                            "timestamp": datetime.now().isoformat()
                        }

                    # We have customer info but DO NOT create cart or add item on the server side. Record pending buy with the provided customer info and ask for confirmation.
                    chosen_size = None
                    chosen_price = None
                    if isinstance(pm, dict) and len(pm) == 1:
                        chosen_size = list(pm.keys())[0]
                        chosen_price = float(list(pm.values())[0])

                    pid = self._normalize_pid(prod.get('product_id') or prod.get('_id'))
                    state['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': 1, 'size': chosen_size, 'unit_price': chosen_price}
                    state['selected_product_id'] = pid
                    _set_state(user_id, state)

                    # Save intent in memory and inform user we are ready for confirmation
                    self.memory.setdefault(user_id, []).append({
                        "role": "bot",
                        "message": f"سجّلت رغبتك بشراء المنتج. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء.",
                        "intent": 'buy',
                        "timestamp": datetime.now().isoformat()
                    })
                    response_text = f"سجّلت رغبتك بشراء المنتج. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                    product_title = (prod.get('title_ar') or prod.get('title') or prod.get('name')) if prod else None
                    data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": chosen_size, "purchase_intent": True, "status": "started", "ask_confirm": True}
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": intent
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Direct buy flow failed: {e}")
                    pass


                # No customer info should be captured at selection time — ask for explicit confirmation only
                # (This replaces earlier behavior that attempted to record name/phone here.)
                try:
                    response_text = "تم بدء إضافة المنتج إلى سلة التسوق بنجاح. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                    product = pending_buy.get('product') if pending_buy else None
                    product_title = (product.get('title_ar') or product.get('title') or product.get('name')) if product else None
                    data = {
                        "action": "start_add_to_cart",
                        "product_title": product_title,
                        "selected_size": pending_buy.get('size'),
                        "purchase_intent": True,
                        "status": "started",
                        "ask_confirm": True
                    }
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": intent
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Buy flow failed: {e}")
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'buy',
                        "intent_confidence": 0.0,
                        "response": "عذراً، لم أتمكن من إضافة المنتج إلى السلة الآن.",
                        "data": None,
                        "suggestions": [],
                        "timestamp": datetime.now().isoformat()
                    }
            # If the user confirmed without providing name/phone, add to cart as guest
            if _is_affirmative(normalized):
                try:
                    prod = pending_buy.get('product') or state.get('last_viewed_product')
                    pinfo = self._get_price_info(prod)
                    pm = pinfo.get('price_map') or {}
                    chosen_size = pending_buy.get('size') or (list(pm.keys())[0] if isinstance(pm, dict) and len(pm) == 1 else None)
                    chosen_price = pending_buy.get('unit_price') or (float(list(pm.values())[0]) if isinstance(pm, dict) and len(pm) == 1 else (pinfo.get('min_price') if pinfo.get('min_price') is not None else pinfo.get('price')))

                    # Do NOT create a guest cart or add item; instead request customer's name/phone to complete the checkout
                    pid = prod.get('product_id') or prod.get('_id')
                    state['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': pending_buy.get('quantity', 1), 'size': chosen_size, 'unit_price': chosen_price}
                    state['selected_product_id'] = pid
                    _set_state(user_id, state)

                    self.memory.setdefault(user_id, []).append({
                        "role": "bot",
                        "message": f"تم بدء إضافة {prod.get('title') or prod.get('name')} إلى سلة التسوق. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء.",
                        "intent": 'buy',
                        "timestamp": datetime.now().isoformat()
                    })

                    response_text = f"تم بدء إضافة {prod.get('title') or prod.get('name')} إلى سلة التسوق بنجاح. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                    product_title = (prod.get('title_ar') or prod.get('title') or prod.get('name')) if prod else None
                    data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": chosen_size, "purchase_intent": True, "status": "started", "ask_confirm": True}
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": intent
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.warning(f"Buy flow failed (guest confirm): {e}")
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'buy',
                        "intent_confidence": 0.0,
                        "response": "عذراً، لم أتمكن من إضافة المنتج إلى السلة الآن.",
                        "data": None,
                        "suggestions": [],
                        "timestamp": datetime.now().isoformat()
                    }

        # --- NEW: allow product selection by number (e.g., '1', '1 تفاصيل', 'رقم 1') ---
        # If the user sends a message that includes a number and we have a prior search context, interpret it as a selection
        if user_id in self.search_context and self.search_context.get(user_id):
            # match either western digits or Arabic-Indic digits, allow optional 'رقم' prefix and extra words after the number
            sel_match = re.search(r'(?:\b|^)\s*(?:رقم\s*)?([0-9\u0660-\u0669]+)', normalized)
            if sel_match:
                sel_raw = sel_match.group(1)
                # translate Arabic-Indic digits to western
                trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                sel_num = int(sel_raw.translate(trans)) if sel_raw else None
                summaries = self.search_context[user_id].get('summaries') if isinstance(self.search_context[user_id], dict) else self.search_context[user_id]
                products = self.search_context[user_id].get('products') if isinstance(self.search_context[user_id], dict) else None

                # If there is an awaiting size selection in state, treat the user's numeric reply as a SIZE CHOICE first
                st_tmp = _get_state(user_id)
                awaiting_tmp = st_tmp.get('awaiting_size_selection') if st_tmp else None
                pending_buy_tmp = st_tmp.get('pending_buy') if st_tmp else None
                if not awaiting_tmp and pending_buy_tmp and pending_buy_tmp.get('awaiting') == 'size_selection':
                    awaiting_tmp = {'product': pending_buy_tmp.get('product'), 'sizes': pending_buy_tmp.get('sizes')}

                if awaiting_tmp and sel_num and 1 <= sel_num <= len(awaiting_tmp.get('sizes') or []):
                    # Handle as size selection
                    sizes = awaiting_tmp.get('sizes') or []
                    idx_size = sel_num - 1
                    if idx_size < 0 or idx_size >= len(sizes):
                        return ({
                            "user_id": user_id,
                            "original_message": message,
                            "normalized_message": normalized,
                            "intent": 'clarify',
                            "intent_confidence": 0.60,
                            "response": f"الاختيار غير صالح للحجم {sel_num}. يرجى كتابة رقم من 1 إلى {len(sizes)}.",
                            "data": None,
                            "suggestions": [],
                            "context_summary": {
                                "turns_count": len(self.memory.get(user_id, [])),
                                "last_activity": datetime.now().isoformat(),
                                "recent_questions": _get_state(user_id).get('recent_questions', [])
                            },
                            "timestamp": datetime.now().isoformat()
                        })

                    size = sizes[idx_size]
                    prod = awaiting_tmp.get('product')
                    # Check availability for the selected size
                    if not self._is_in_stock(prod, size=size):
                        response_text = f"عذراً، الحجم {size} غير متوفر حالياً. يرجى اختيار حجم آخر من القائمة أو اكتب 'إلغاء'."
                        try:
                            _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                        except Exception:
                            pass
                        return {
                            "user_id": user_id,
                            "original_message": message,
                            "normalized_message": normalized,
                            "intent": 'clarify',
                            "intent_confidence": 0.60,
                            "response": response_text,
                            "data": {"sizes": sizes},
                            "suggestions": [],
                            "timestamp": datetime.now().isoformat()
                        }

                    # Determine unit price robustly
                    p_info_sel = self._get_price_info(prod)
                    raw_val = p_info_sel.get('price_map', {}).get(size)
                    if isinstance(raw_val, dict):
                        raw_val = raw_val.get('price') or raw_val.get('value') or raw_val.get('amount')
                    try:
                        unit_price = float(raw_val) if raw_val is not None else float(p_info_sel.get('min_price') or 0)
                    except Exception:
                        try:
                            unit_price = float(p_info_sel.get('min_price') or prod.get('price') or 0)
                        except Exception:
                            unit_price = 0.0

                    # set pending buy to await confirmation (no name/phone required)
                    st = _get_state(user_id)
                    st.pop('awaiting_size_selection', None)
                    st['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': 1, 'size': size, 'unit_price': unit_price}
                    st['selected_product_id'] = prod.get('product_id') or prod.get('_id')
                    _set_state(user_id, st)
                    response_text = f"اخترت الحجم {size} بسعر {unit_price} {prod.get('currency','USD')}، هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                    try:
                        _log_bot_turn(user_id, response_text, 'buy', {'size': size, 'unit_price': unit_price})
                    except Exception:
                        pass
                    product_title = (prod.get('title_ar') or prod.get('title') or prod.get('name')) if prod else None
                    data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": size, "purchase_intent": True, "status": "started", "ask_confirm": True}
                    return {"user_id": user_id, "original_message": message, "normalized_message": normalized, "intent": intent, "intent_confidence": intent_result['confidence'], "response": response_text, "data": data, "suggestions": self._get_suggestions(intent)}

                if sel_num and summaries and 1 <= sel_num <= len(summaries):
                    idx = sel_num - 1
                    chosen_summary = summaries[idx]
                    # Try to find full product object
                    chosen_prod = None
                    if products:
                        try:
                            chosen_prod = products[idx]
                        except Exception:
                            chosen_prod = None
                    if not chosen_prod:
                        # Fallback: fetch by product_id
                        pid = chosen_summary.get('product_id')
                        chosen_prod = mongo_service.get_product_by_id(pid)

                    if chosen_prod:
                        # If there's an existing awaiting size selection, treat this numeric selection as a SIZE CHOICE
                        st = _get_state(user_id)
                        awaiting = st.get('awaiting_size_selection') if st else None
                        pending_buy = st.get('pending_buy') if st else None
                        if not awaiting and pending_buy and pending_buy.get('awaiting') == 'size_selection':
                            awaiting = {'product': pending_buy.get('product'), 'sizes': pending_buy.get('sizes')}
                        if awaiting:
                            # Interpret the number as a size selection for a pending product
                            sizes = awaiting.get('sizes') or []
                            idx_size = sel_num - 1
                            if idx_size < 0 or idx_size >= len(sizes):
                                return ({
                                    "user_id": user_id,
                                    "original_message": message,
                                    "normalized_message": normalized,
                                    "intent": 'clarify',
                                    "intent_confidence": 0.60,
                                    "response": f"الاختيار غير صالح للحجم {sel_num}. يرجى كتابة رقم من 1 إلى {len(sizes)}.",
                                    "data": None,
                                    "suggestions": [],
                                    "context_summary": {
                                        "turns_count": len(self.memory.get(user_id, [])),
                                        "last_activity": datetime.now().isoformat(),
                                        "recent_questions": _get_state(user_id).get('recent_questions', [])
                                    },
                                    "timestamp": datetime.now().isoformat()
                                } )
                            size = sizes[idx_size]
                            prod = awaiting.get('product')
                            # Check availability for the selected size
                            if not self._is_in_stock(prod, size=size):
                                response_text = f"عذراً، الحجم {size} غير متوفر حالياً. يرجى اختيار حجم آخر من القائمة أو اكتب 'إلغاء'."
                                try:
                                    _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                                except Exception:
                                    pass
                                return {
                                    "user_id": user_id,
                                    "original_message": message,
                                    "normalized_message": normalized,
                                    "intent": 'clarify',
                                    "intent_confidence": 0.60,
                                    "response": response_text,
                                    "data": {"sizes": sizes},
                                    "suggestions": [],
                                    "timestamp": datetime.now().isoformat()
                                }

                            # Determine unit price robustly using normalized price info
                            p_info_sel = self._get_price_info(prod)
                            raw_val = p_info_sel.get('price_map', {}).get(size)
                            if isinstance(raw_val, dict):
                                raw_val = raw_val.get('price') or raw_val.get('value') or raw_val.get('amount')
                            try:
                                unit_price = float(raw_val) if raw_val is not None else float(p_info_sel.get('min_price') or 0)
                            except Exception:
                                try:
                                    unit_price = float(p_info_sel.get('min_price') or prod.get('price') or 0)
                                except Exception:
                                    unit_price = 0.0
                            # set pending buy to await confirmation (no name/phone required)
                            st.pop('awaiting_size_selection', None)
                            st['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': 1, 'size': size, 'unit_price': unit_price}
                            st['selected_product_id'] = self._normalize_pid(prod.get('product_id') or prod.get('_id'))
                            _set_state(user_id, st)
                            response_text = f"اخترت الحجم {size} بسعر {unit_price} {prod.get('currency','USD')}، هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                            intent = 'buy'
                            intent_result['confidence'] = 0.95
                            self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                            try:
                                _log_bot_turn(user_id, response_text, 'buy', {'size': size, 'unit_price': unit_price})
                            except Exception:
                                pass
                            pid = self._normalize_pid(prod.get('product_id') or prod.get('_id'))
                            product_title = (prod.get('title_ar') or prod.get('title') or prod.get('name')) if prod else None
                            data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": size, "purchase_intent": True, "status": "started", "ask_confirm": True}
                            return {"user_id": user_id, "original_message": message, "normalized_message": normalized, "intent": intent, "intent_confidence": intent_result['confidence'], "response": response_text, "data": data, "suggestions": self._get_suggestions(intent)}

                        # save last viewed product to state for buy flow or direct actions
                        st = _get_state(user_id)
                        st['last_viewed_product'] = chosen_prod
                        st['selected_product_id'] = self._normalize_pid(chosen_prod)
                        logger.debug('selected_product_id set to %s for user %s', st['selected_product_id'], user_id)
                        _set_state(user_id, st)

                        # Build product detail response
                        title = chosen_prod.get('title_ar') or chosen_prod.get('title') or chosen_prod.get('name')
                        desc = chosen_prod.get('description_ar') or chosen_prod.get('description') or ''
                        # Build price display: if multiple sizes, list them line-by-line; else show single price
                        p_info = self._get_price_info(chosen_prod)
                        pm = p_info.get('price_map') or {}
                        if isinstance(pm, dict) and pm:
                            lines = []
                            for k, v in pm.items():
                                # support variant records where v may be a dict like {'price': x, 'stock': y}
                                if isinstance(v, dict):
                                    pv = v.get('price') or v.get('value') or v.get('amount')
                                else:
                                    pv = v
                                price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                                lines.append(f"- {k}: {price_str} {chosen_prod.get('currency','') or ''}")
                            prices = "\n" + "\n".join(lines)
                        else:
                            single_price = p_info.get('min_price')
                            prices = f"{single_price:.1f}" if isinstance(single_price, (int, float)) else str(single_price or 'N/A')
                        availability = "متوفر" if self._is_in_stock(chosen_prod) else "غير متوفر"

                        # If the user's message expresses an intent to buy (e.g., 'اشتري 1'), start buy flow
                        wants_to_buy = False
                        try:
                            buy_kws = getattr(self.intent_detector, 'buy_keywords', [])
                            if any(self.intent_detector._contains_keyword(normalized.lower(), kw) for kw in buy_kws) or re.search(r'\bاشتري\b', normalized.lower()):
                                wants_to_buy = True
                        except Exception:
                            wants_to_buy = False

                        if wants_to_buy:
                            # If product has multiple sizes, prompt for size selection
                            if isinstance(pm, dict) and len(pm) > 1:
                                sizes = list(pm.keys())
                                choices = []
                                for i, s in enumerate(sizes):
                                    val = pm.get(s)
                                    price_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val or 'N/A')
                                    # If product-level inStock is present, prefer it; otherwise check variant stock if available
                                    avail = "متوفر" if self._is_in_stock(chosen_prod, size=s) else "غير متوفر"
                                    choices.append(f"{i+1}. {s} - {price_str} {chosen_prod.get('currency','') or ''} - {avail}")
                                st['awaiting_size_selection'] = {'product': chosen_prod, 'sizes': sizes}
                                pid = chosen_prod.get('product_id') or chosen_prod.get('_id')
                                st['pending_buy'] = {'product': {'product_id': pid, 'name': title}, 'awaiting': 'size_selection'}
                                st['selected_product_id'] = pid
                                _set_state(user_id, st)
                                try:
                                    _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                                except Exception:
                                    pass
                                response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                                return {
                                    "user_id": user_id,
                                    "original_message": message,
                                    "normalized_message": normalized,
                                    "intent": 'clarify',
                                    "intent_confidence": 0.85,
                                    "response": response_text,
                                    "data": {"sizes": sizes},
                                    "suggestions": ["1", "2"],
                                    "context_summary": {
                                        "turns_count": len(self.memory.get(user_id, [])),
                                        "last_activity": datetime.now().isoformat(),
                                        "recent_questions": _get_state(user_id).get('recent_questions', [])
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                            # Otherwise, check if product has multiple sizes and require size selection first
                            p_info = self._get_price_info(chosen_prod)
                            sizes = p_info.get('sizes') or []
                            if sizes and len(sizes) > 1:
                                # prompt for size selection
                                choices = []
                                for i, s in enumerate(sizes):
                                    val = p_info['price_map'].get(s)
                                    price_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val or 'N/A')
                                    choices.append(f"{i+1}. {s} - {price_str} {chosen_prod.get('currency','') or ''}")
                                st['awaiting_size_selection'] = {'product': chosen_prod, 'sizes': sizes}
                                pid = chosen_prod.get('product_id') or chosen_prod.get('_id')
                                st['pending_buy'] = {'product': {'product_id': pid, 'name': title}, 'awaiting': 'size_selection'}
                                st['selected_product_id'] = pid
                                _set_state(user_id, st)
                                response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                                try:
                                    _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                                except Exception:
                                    pass
                                return {
                                    "user_id": user_id,
                                    "original_message": message,
                                    "normalized_message": normalized,
                                    "intent": 'clarify',
                                    "intent_confidence": 0.85,
                                    "response": response_text,
                                    "data": {"sizes": sizes},
                                    "suggestions": ["1", "2"],
                                    "context_summary": {
                                        "turns_count": len(self.memory.get(user_id, [])),
                                        "last_activity": datetime.now().isoformat(),
                                        "recent_questions": _get_state(user_id).get('recent_questions', [])
                                    },
                                    "timestamp": datetime.now().isoformat()
                                }
                            # Otherwise, set pending_buy and ask for customer info
                            pid = chosen_prod.get('product_id') or chosen_prod.get('_id')
                            st['pending_buy'] = {'product': {'product_id': pid, 'name': title}, 'awaiting': 'confirm_checkout', 'quantity': 1}
                            st['selected_product_id'] = pid
                            _set_state(user_id, st)
                            response_text = f"لقد اخترت '{title}'. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                            product_title = title
                            data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": None, "purchase_intent": True, "status": "started", "ask_confirm": True}
                            return self._format_response(user_id, message, normalized, 'buy', 0.95, response_text, data=data, suggestions=["نعم", "لا"], context_summary={"turns_count": len(self.memory.get(user_id, [])), "last_activity": datetime.now().isoformat(), "recent_questions": _get_state(user_id).get('recent_questions', [])})

                        # Default: return detail view
                        resp = f"تفاصيل المنتج ({sel_num}):\n{title}\nالسعر: {prices} {chosen_prod.get('currency','') or ''}\nالتوفر: {availability}\n\n{desc}\n\nلإضافة هذا المنتج إلى السلة اكتب 'اشتري {sel_num}'   ."
                        return self._format_response(user_id, message, normalized, 'detail', 0.95, resp, data={"product": chosen_prod}, suggestions=["اشتري {}".format(sel_num), "شاهد منتجات مشابهة"], context_summary={"turns_count": len(self.memory.get(user_id, [])), "last_activity": datetime.now().isoformat(), "recent_questions": _get_state(user_id).get('recent_questions', [])})
                # If number found but out of range, clarify
                if sel_num:
                    return ({
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'clarify',
                        "intent_confidence": 0.60,
                        "response": f"لم أجد خيارًا بالرقم {sel_num}. يرجى كتابة رقم من 1 إلى {len(summaries) if summaries else 0}.",
                        "data": None,
                        "suggestions": [],
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "recent_questions": self.user_state.get(user_id, {}).get('recent_questions', [])
                        },
                        "timestamp": datetime.now().isoformat()
                    })


        # Confirmation step: user confirms checkout
        if pending_buy and pending_buy.get('awaiting') == 'confirm_checkout':
            # Ensure a size was selected when product has multiple sizes
            last_cart = state.get('last_cart') or {}
            selected_pid = state.get('selected_product_id')
            size_selected = False
            try:
                # If the user already selected a size earlier in the pending buy, honor it
                if pending_buy and pending_buy.get('size'):
                    size_selected = True
                elif last_cart and selected_pid:
                    for it in last_cart.get('items', []):
                        if it.get('product_id') == selected_pid and it.get('size'):
                            size_selected = True
                            break
            except Exception:
                size_selected = False

            # If no size selected but product has multiple sizes, prompt for size first
            if not size_selected:
                prod = None
                if pending_buy and pending_buy.get('product'):
                    prod = pending_buy.get('product')
                else:
                    prod = state.get('last_viewed_product')
                pinfo = self._get_price_info(prod)
                pm = pinfo.get('price_map') or {}
                if isinstance(pm, dict) and len(pm) > 1:
                    sizes = list(pm.keys())
                    choices = []
                    for i, s in enumerate(sizes):
                        val = pm.get(s)
                        if isinstance(val, dict):
                            pv = val.get('price') or val.get('value') or val.get('amount')
                        else:
                            pv = val
                        price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                        choices.append(f"{i+1}. {s} - {price_str} {prod.get('currency','') or ''}")
                    # set awaiting size selection
                    state['awaiting_size_selection'] = {'product': prod, 'sizes': sizes}
                    state['pending_buy'] = {'product': prod, 'awaiting': 'size_selection'}
                    _set_state(user_id, state)
                    response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد قبل تأكيد الطلب:\n" + "\n".join(choices)
                    try:
                        _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                    except Exception:
                        pass
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": 'clarify',
                        "intent_confidence": 0.80,
                        "response": response_text,
                        "data": {"sizes": sizes},
                        "suggestions": [],
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": 'clarify'
                        },
                        "timestamp": datetime.now().isoformat()
                    }

            # Affirmative -> record confirmation and return payload (no server-side add)
            if _is_affirmative(normalized):
                try:
                    prod = pending_buy.get('product') if pending_buy else state.get('last_viewed_product')
                    pid = state.get('selected_product_id') or (self._normalize_pid(prod) if prod else None)
                    logger.debug('Confirming buy for user %s: pid=%s pending_buy=%s', user_id, pid, state.get('pending_buy'))
                    chosen_size = pending_buy.get('size') if pending_buy else None
                    # Clear pending buy since the user confirmed
                    state.pop('pending_buy', None)
                    self._set_state(user_id, state)

                    response_text = "تم بدء طلب إضافة المنتج إلى سلة التسوق بنجاح."
                    product_title = (prod.get('title_ar') or prod.get('title') or prod.get('name')) if prod else None
                    data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": chosen_size, "purchase_intent": True, "status": "started", "ask_confirm": False}
                    intent = 'buy'
                    intent_result['confidence'] = 0.95
                    self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                    try:
                        self._log_bot_turn(user_id, response_text, 'buy', {'product_title': product_title, 'selected_size': chosen_size})
                    except Exception:
                        pass
                    return self._format_response(user_id, message, normalized, intent, intent_result["confidence"], response_text, data=data, suggestions=[])
                except Exception as e:
                    logger.warning(f"Confirmation handling failed: {e}")
                    return {"user_id": user_id, "original_message": message, "normalized_message": normalized, "intent": 'buy', "intent_confidence": 0.0, "response": "عذراً، حدث خطأ عند معالجة التأكيد.", "data": None, "suggestions": []}


            # Negative -> cancel pending checkout
            if normalized.strip() in ['لا', 'الغاء', 'إلغاء', 'الغاء الطلب']:
                state.pop('pending_buy', None)
                state.pop('selected_product_id', None)
                self._set_state(user_id, state)
                response_text = "تم إلغاء تأكيد الطلب. ما الذي تود فعله الآن؟"
                intent = 'buy'
                intent_result['confidence'] = 0.85
                self.memory[user_id].append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                try:
                    _finalize_training_session(user_id, 'purchase_cancelled')
                except Exception:
                    pass
                return {
                    "user_id": user_id,
                    "original_message": message,
                    "normalized_message": normalized,
                    "intent": intent,
                    "intent_confidence": intent_result["confidence"],
                    "response": response_text,
                    "data": None,
                    "suggestions": self._get_suggestions(intent),
                    "context_summary": {
                        "turns_count": len(self.memory.get(user_id, [])),
                        "last_activity": datetime.now().isoformat(),
                        "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                        "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                        "last_intent": intent
                    },
                    "timestamp": datetime.now().isoformat()
                }
            # otherwise keep waiting for explicit confirmation

        # If the user explicitly ends the session, clear shopping cart and end the session
        if intent == 'end_session':
            state = self._get_state(user_id)
            cart_id = None
            if state:
                cart_id = (state.get('last_cart') or {}).get('cart_id') or (state.get('pending_buy') or {}).get('cart_id')
            try:
                if cart_id:
                    mongo_service.delete_cart(cart_id)
                # finalize training session and clear session state and memory
                try:
                    _finalize_training_session(user_id, 'session_ended')
                except Exception:
                    pass
                self.clear_conversation(user_id)
                self._clear_state(user_id)
                response_text = "تم إنهاء الجلسة وحذف سلة التسوق الخاصة بك. إذا رغبت في الشراء لاحقًا فأنا هنا للمساعدة. مع السلامة 👋"
                data = {"set_cookie": {"name": "cart_id", "value": "", "max_age": 0}}
                intent_result['confidence'] = 0.99
                intent = 'end_session'
                # record bot message and return
                self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                try:
                    _log_bot_turn(user_id, response_text, intent)
                except Exception:
                    pass
                return {
                    "user_id": user_id,
                    "original_message": message,
                    "normalized_message": normalized,
                    "intent": intent,
                    "intent_confidence": intent_result["confidence"],
                    "response": response_text,
                    "data": data,
                    "suggestions": [],
                    "context_summary": {
                        "turns_count": 0,
                        "last_activity": datetime.now().isoformat(),
                        "user_messages": 0,
                        "bot_messages": 0,
                        "last_intent": intent
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"End session cleanup failed: {e}")
                return ("عذراً، حدث خطأ أثناء إنهاء الجلسة. حاول مرة أخرى لاحقًا.", None)

        # If the user replies with a pure numeric choice we may be selecting a product
        # from recent results, or choosing a size when awaiting size selection.
        if re.match(r'^\s*[0-9٠١٢٣٤٥٦٧٨٩]+[\.)]?\s*$', normalized):
            state = self._get_state(user_id)
            awaiting = state.get('awaiting_size_selection')
            # Also support a pending_buy waiting for size (awaiting == 'size_selection')
            pending_buy = state.get('pending_buy') if state else None
            if not awaiting and pending_buy and pending_buy.get('awaiting') == 'size_selection':
                awaiting = {'product': pending_buy.get('product'), 'sizes': pending_buy.get('sizes')}

            if awaiting:
                # Parse index (support Arabic-Indic digits)
                s = re.sub(r'[^0-9٠١٢٣٤٥٦٧٨٩]', '', normalized)
                trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
                s = s.translate(trans)
                try:
                    idx = int(s) - 1
                except Exception:
                    idx = -1
                sizes = awaiting.get('sizes') or []
                if idx < 0 or idx >= len(sizes):
                    response_text = "الاختيار غير صالح، يرجى إدخال رقم صحيح من القائمة." 
                    return {"user_id": user_id, "response": response_text, "intent": 'clarify', "intent_confidence": 0.60}
                size = sizes[idx]
                prod = awaiting.get('product')
                # Determine unit price robustly using normalized price info
                p_info_sel = self._get_price_info(prod)
                raw_val = p_info_sel.get('price_map', {}).get(size)
                if isinstance(raw_val, dict):
                    raw_val = raw_val.get('price') or raw_val.get('value') or raw_val.get('amount')
                try:
                    unit_price = float(raw_val) if raw_val is not None else float(p_info_sel.get('min_price') or 0)
                except Exception:
                    try:
                        unit_price = float(p_info_sel.get('min_price') or prod.get('price') or 0)
                    except Exception:
                        unit_price = 0.0
                # set pending buy to await confirmation (no name/phone required)
                state.pop('awaiting_size_selection', None)
                state['pending_buy'] = {'product': prod, 'awaiting': 'confirm_checkout', 'quantity': 1, 'size': size, 'unit_price': unit_price}
                state['selected_product_id'] = prod.get('product_id') or prod.get('_id')
                _set_state(user_id, state)
                response_text = f"اخترت الحجم {size} بسعر {unit_price} {prod.get('currency','USD')}، هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                intent = 'buy'
                intent_result['confidence'] = 0.95
                self.memory.setdefault(user_id, []).append({"role": "bot", "message": response_text, "intent": intent, "timestamp": datetime.now().isoformat()})
                try:
                    _log_bot_turn(user_id, response_text, 'buy', {'size': size, 'unit_price': unit_price})
                except Exception:
                    pass
                return self._format_response(user_id, message, normalized, intent, intent_result['confidence'], response_text, data={"size": size, "unit_price": unit_price}, suggestions=self._get_suggestions(intent))

            # Fallback to interpreting as product selection from search results
            if self.search_context.get(user_id):
                # Forward numeric selection to detail handler
                response_text, data = self._handle_detail_request(user_id, normalized)
                intent = 'detail'
                intent_result['confidence'] = 0.95
                # Store bot response in memory and return
                self.memory[user_id].append({
                    "role": "bot",
                    "message": response_text,
                    "intent": intent,
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "user_id": user_id,
                    "original_message": message,
                    "normalized_message": normalized,
                    "intent": intent,
                    "intent_confidence": intent_result["confidence"],
                    "response": response_text,
                    "data": data,
                    "suggestions": self._get_suggestions(intent),
                    "context_summary": {
                        "turns_count": len(self.memory.get(user_id, [])),
                        "last_activity": datetime.now().isoformat(),
                        "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                        "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                        "last_intent": intent
                    },
                    "timestamp": datetime.now().isoformat()
                }
        # If the user replied with a short affirmative and we have recent search context,
        # interpret as a request for details of the item (single or ask to clarify)
        # Special case: if user says 'نعم' and we have a last viewed product, start buy flow
        if _is_affirmative(normalized):
            # If user already viewed a product and hasn't started buy, treat 'نعم' as 'buy this'
            state = self._get_state(user_id)
            last_viewed = state.get('last_viewed_product') if state else None
            if last_viewed and not state.get('pending_buy'):
                pm = last_viewed.get('price_map') or {}
                # If product has multiple sizes, ask for size first
                if isinstance(pm, dict) and len(pm) > 1:
                    sizes = list(pm.keys())
                    choices = []
                    for i, s in enumerate(sizes):
                        val = pm.get(s)
                        price_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val or 'N/A')
                        choices.append(f"{i+1}. {s} - {price_str} {last_viewed.get('currency','USD')}")
                    state['awaiting_size_selection'] = {'product': last_viewed, 'sizes': sizes}
                    self._set_state(user_id, state)
                    response_text = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                    intent = 'clarify'
                    intent_result['confidence'] = 0.80
                    self.memory[user_id].append({
                        "role": "bot",
                        "message": response_text,
                        "intent": intent,
                        "timestamp": datetime.now().isoformat()
                    })
                    try:
                        _log_bot_turn(user_id, response_text, 'clarify', {'sizes': sizes})
                    except Exception:
                        pass
                    data = {"sizes": sizes}
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": [],
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": 'clarify'
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                # If product has a single size or no size map, proceed to ask for customer info
                size = None
                unit_price = None
                if isinstance(pm, dict) and len(pm) == 1:
                    size = list(pm.keys())[0]
                    try:
                        unit_price = float(pm[size])
                    except Exception:
                        unit_price = None
                state['pending_buy'] = {'product': last_viewed, 'awaiting': 'confirm_checkout', 'quantity': 1}
                if size:
                    state['pending_buy'].update({'size': size, 'unit_price': unit_price})
                state['selected_product_id'] = (last_viewed or {}).get('product_id') or (last_viewed or {}).get('_id')
                _set_state(user_id, state)
                try:
                    _log_bot_turn(user_id, response_text, 'buy')
                except Exception:
                    pass
                response_text = "لتأكيد الشراء، يرجى تزويدي باسمك الكامل ورقم هاتفك (مثال: أحمد, 0501234567)"
                intent = 'buy'
                intent_result['confidence'] = 0.90
                self.memory[user_id].append({
                    "role": "bot",
                    "message": response_text,
                    "intent": intent,
                    "timestamp": datetime.now().isoformat()
                })
                data = None
                return {
                    "user_id": user_id,
                    "original_message": message,
                    "normalized_message": normalized,
                    "intent": intent,
                    "intent_confidence": intent_result["confidence"],
                    "response": response_text,
                    "data": data,
                    "suggestions": self._get_suggestions(intent),
                    "context_summary": {
                        "turns_count": len(self.memory.get(user_id, [])),
                        "last_activity": datetime.now().isoformat(),
                        "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                        "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                        "last_intent": intent
                    },
                    "timestamp": datetime.now().isoformat()
                }

            if self.search_context.get(user_id):
                ctx = self.search_context.get(user_id)
                # normalize context to a list of summaries when it may be stored as a dict
                if isinstance(ctx, dict):
                    last_summaries = ctx.get('summaries') or ctx.get('products') or []
                else:
                    last_summaries = ctx or []

                if len(last_summaries) == 1:
                    # User likely wants details for the only returned product
                    response_text, data = self._handle_detail_request(user_id, '1')
                    intent = 'detail'
                    intent_result['confidence'] = 0.90
                    # Store bot response in memory and return
                    self.memory[user_id].append({
                        "role": "bot",
                        "message": response_text,
                        "intent": intent,
                        "timestamp": datetime.now().isoformat()
                    })
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": intent
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Multiple results - ask user to choose which one
                    choices = [f"{i+1}. {s.get('name', s.get('title',''))}" for i, s in enumerate(last_summaries[:5])]
                    prompt = "لقد وجدت عدة منتجات، أي واحد تقصد؟\n" + "\n".join(choices)
                    response_text = prompt
                    data = None
                    intent = 'clarify'
                    intent_result['confidence'] = 0.80
                    self.memory[user_id].append({
                        "role": "bot",
                        "message": response_text,
                        "intent": intent,
                        "timestamp": datetime.now().isoformat()
                    })
                    return {
                        "user_id": user_id,
                        "original_message": message,
                        "normalized_message": normalized,
                        "intent": intent,
                        "intent_confidence": intent_result["confidence"],
                        "response": response_text,
                        "data": data,
                        "suggestions": self._get_suggestions(intent),
                        "context_summary": {
                            "turns_count": len(self.memory.get(user_id, [])),
                            "last_activity": datetime.now().isoformat(),
                            "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                            "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                            "last_intent": intent
                        },
                        "timestamp": datetime.now().isoformat()
                    }

        # Quick handling: if user asks for product details and we have a recent search,
        # handle detail requests directly (e.g., 'نعم تفاصيل ...', 'أريد تفاصيل', 'تفاصيل')
        detail_keywords = ['تفاصيل', 'تفصيل', 'مواصفات', 'تفاصيل المنتج', 'أريد تفاصيل', 'نعم تفاصيل', 'اعطني تفاصيل', 'أعطني تفاصيل']
        if any(k in normalized for k in detail_keywords):
            if self.search_context.get(user_id):
                response_text, data = self._handle_detail_request(user_id, normalized)
                intent = 'detail'
                intent_result['confidence'] = 0.90
                # Store bot response in memory below and return
                self.memory[user_id].append({
                    "role": "bot",
                    "message": response_text,
                    "intent": intent,
                    "timestamp": datetime.now().isoformat()
                })
                return {
                    "user_id": user_id,
                    "original_message": message,
                    "normalized_message": normalized,
                    "intent": intent,
                    "intent_confidence": intent_result["confidence"],
                    "response": response_text,
                    "data": data,
                    "suggestions": self._get_suggestions(intent),
                    "context_summary": {
                        "turns_count": len(self.memory.get(user_id, [])),
                        "last_activity": datetime.now().isoformat(),
                        "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                        "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                        "last_intent": intent
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # No recent search context
                intent = 'fallback'
                intent_result['confidence'] = 0.60
                # continue to normal fallback handling

        # If intent is price, handle price queries and ranges
        if intent == 'price':
            response_text, data = self._handle_price_intent(normalized)
            # keep last search context if results exist
            if data and data.get('summaries'):
                self.search_context[user_id] = data.get('summaries')
            # Store bot response in memory and return
            self.memory[user_id].append({
                "role": "bot",
                "message": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "user_id": user_id,
                "original_message": message,
                "normalized_message": normalized,
                "intent": intent,
                "intent_confidence": intent_result["confidence"],
                "response": response_text,
                "data": data,
                "suggestions": self._get_suggestions(intent),
                "context_summary": {
                    "turns_count": len(self.memory.get(user_id, [])),
                    "last_activity": datetime.now().isoformat(),
                    "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                    "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                    "last_intent": intent
                },
                "timestamp": datetime.now().isoformat()
            }

        # If intent is best, handle best-product requests
        if intent == 'best':
            response_text, data = self._handle_best_intent(normalized)
            if data and data.get('summaries'):
                self.search_context[user_id] = data.get('summaries')
            self.memory[user_id].append({
                "role": "bot",
                "message": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "user_id": user_id,
                "original_message": message,
                "normalized_message": normalized,
                "intent": intent,
                "intent_confidence": intent_result["confidence"],
                "response": response_text,
                "data": data,
                "suggestions": self._get_suggestions(intent),
                "context_summary": {
                    "turns_count": len(self.memory.get(user_id, [])),
                    "last_activity": datetime.now().isoformat(),
                    "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                    "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                    "last_intent": intent
                },
                "timestamp": datetime.now().isoformat()
            }

        # If intent is buy, handle buy/add-to-cart requests
        if intent == 'buy':
            response_text, data = self._handle_buy_intent(normalized, user_id)
            # if _handle_buy_intent returns summaries (on product search) keep context
            if data and data.get('cart'):
                # Optionally keep cart summary in search_context for follow-ups
                self.search_context[user_id] = data.get('cart').get('items', [])
            self.memory[user_id].append({
                "role": "bot",
                "message": response_text,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "user_id": user_id,
                "original_message": message,
                "normalized_message": normalized,
                "intent": intent,
                "intent_confidence": intent_result["confidence"],
                "response": response_text,
                "data": data,
                "suggestions": self._get_suggestions(intent),
                "context_summary": {
                    "turns_count": len(self.memory.get(user_id, [])),
                    "last_activity": datetime.now().isoformat(),
                    "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                    "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                    "last_intent": intent
                },
                "timestamp": datetime.now().isoformat()
            }

        # If intent is fallback, try to infer search intent from keywords or
        # presence of brand/category in the query (e.g., 'منتجات', or brand name)
        if intent == 'fallback':
            simple = self._simplify_text(normalized)
            # If user mentions buying or add-to-cart verbs, prefer buy intent
            try:
                buy_kws = getattr(self.intent_detector, 'buy_keywords', [])
                if any(self.intent_detector._contains_keyword(normalized.lower(), kw) or kw in normalized.lower() for kw in buy_kws):
                    intent = 'buy'
                    intent_result['confidence'] = 0.85
                elif 'منتج' in simple or 'منتجات' in simple:
                    intent = 'search'
                    intent_result['confidence'] = 0.80
                else:
                    filters = self._extract_filters_from_query(normalized)
                    if filters.get('brand') or filters.get('category'):
                        intent = 'search'
                        intent_result['confidence'] = 0.80
                    else:
                        # If the user's short text directly matches products in DB, treat as search
                        try:
                            quick = mongo_service.search_products(normalized, limit=5)
                            if quick and len(quick) > 0:
                                intent = 'search'
                                intent_result['confidence'] = 0.75
                        except Exception:
                            pass
            except Exception:
                # fallback behavior
                if 'منتج' in simple or 'منتجات' in simple:
                    intent = 'search'
                    intent_result['confidence'] = 0.80

        # Handle search intent specially to return product results when possible
        if intent == 'search':
            response_text, data = self._handle_search_intent(normalized)
            # Keep last search summaries and product objects so follow-ups like 'تفاصيل ...' can refer to them
            if data and data.get('summaries'):
                # store both summaries and the raw products list to support selection by number
                self.search_context[user_id] = {"summaries": data.get('summaries'), "products": data.get('products')}
                # Also update recent questions memory (keep up to last 3 user queries)
                rs = _get_state(user_id)
                recent = rs.get('recent_questions', [])
                recent.append(normalized)
                # keep last 3
                recent = recent[-3:]
                rs['recent_questions'] = recent
                _set_state(user_id, rs)
        else:
            response_text = self._generate_response(intent, normalized)
            data = None
        
        # Store bot response in memory
        self.memory[user_id].append({
            "role": "bot",
            "message": response_text,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        
        # Return the result in the format your UI expects
        return {
            "user_id": user_id,
            "original_message": message,
            "normalized_message": normalized,
            "intent": intent,
            "intent_confidence": intent_result["confidence"],
            "response": response_text,
            "data": data,
            "suggestions": self._get_suggestions(intent),
            "context_summary": {
                "turns_count": len(self.memory.get(user_id, [])),
                "last_activity": datetime.now().isoformat(),
                "user_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "user"]),
                "bot_messages": len([m for m in self.memory.get(user_id, []) if m["role"] == "bot"]),
                "last_intent": intent
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_response(self, intent: str, query: str = "") -> str:
        """Generate Arabic response based on intent"""
        
        responses = {
            "greeting": "اهلاً وسهلاً! أنا مساعدك الذكي للتسوق. كيف يمكنني مساعدتك اليوم؟",
            
            "search": f"أبحث لك عن '{query}' في منتجاتنا...\n\nلدي العديد من المنتجات مثل:\n• كريم مرطب للوجه\n• شامبو للشعر الجاف\n• أحمر شفاه مات\n• عطور فاخرة\n\nماذا تريد بالضبط؟",
            
            "price": f"أحضر لك سعر '{query}'...\n\nللعثور على السعر الدقيق، يرجى كتابة اسم المنتج كاملاً مثل:\n'سعر كريم نيڤيا'\n'بكم أحمر شفاه ماك'",
            
            "offers": "🎯 لدينا هذه العروض الحالية:\n\n1. تخفيضات العناية بالبشرة - خصم 20%\n2. عرض العطور الفاخرة - خصم 15%\n3. منتجات جديدة بأسعار خاصة\n\nهل تريد معرفة المزيد عن عرض معين؟",
            
            "help": "🤖 يمكنني مساعدتك في:\n\n1. البحث عن منتجات التجميل والعناية\n2. معرفة الأسعار والتوفر\n3. عرض العروض والخصومات الحالية\n4. الإجابة على أسئلتك\n\nما الذي تبحث عنه؟",

            "closing": "شكرًا لك! إذا احتجت أي شيء آخر فأنا هنا للمساعدة. مع السلامة 👋",
            
            "fallback": "عذراً، لم أفهم سؤالك بالكامل. يمكنني مساعدتك في:\n- البحث عن منتجات\n- معرفة الأسعار\n- عرض العروض\n\nهل يمكنك إعادة صياغة سؤالك؟"
        }
        
        return responses.get(intent, responses["fallback"])
    
    def _get_suggestions(self, intent: str) -> List[str]:
        """Get suggested next actions"""
        suggestions_map = {
            "greeting": ["البحث عن منتج", "عرض العروض", "طلب المساعدة"],
            "search": ["ابحث عن كريم مرطب", "عندكم شامبو", "وين أحمر شفاه"],
            "price": ["سعر كريم نيڤيا", "بكم أحمر شفاه", "كم سعر العطر"],
            "offers": ["تفاصيل عرض العناية", "هل هناك عروض جديدة؟", "شروط العروض"],
            "help": ["البحث عن منتج", "عرض العروض", "الاستفسار عن سعر"],
            "closing": ["إلى اللقاء", "البحث عن منتج", "العودة للصفحة الرئيسية"],
            "fallback": ["البحث عن منتج", "عرض العروض", "طلب المساعدة"]
        }
        return suggestions_map.get(intent, [])

    def _format_response(self, user_id: str, message: str, normalized: str, intent: str, intent_confidence: float, response_text: str, data: Optional[Dict[str, Any]] = None, suggestions: Optional[List[str]] = None, context_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standardize and log the full bot response JSON returned to the UI."""
        if suggestions is None:
            suggestions = self._get_suggestions(intent)
        if context_summary is None:
            context_summary = {
                "turns_count": len(self.memory.get(user_id, [])),
                "last_activity": datetime.now().isoformat(),
                "user_messages": len([m for m in self.memory.get(user_id, []) if m.get("role") == "user"]),
                "bot_messages": len([m for m in self.memory.get(user_id, []) if m.get("role") == "bot"]),
                "last_intent": intent
            }
        resp = {
            "user_id": user_id,
            "original_message": message,
            "normalized_message": normalized,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "response": response_text,
            "data": data,
            "suggestions": suggestions,
            "context_summary": context_summary,
            "timestamp": datetime.now().isoformat()
        }
        try:
            logger.info("Bot response: %s", json.dumps(resp, ensure_ascii=False))
        except Exception:
            logger.debug("Bot response (raw): %s", resp)
        return resp
    
    def _simplify_text(self, s: str) -> str:
        """Simplify Arabic text for more robust substring matching.
        Normalizes common letter variants, removes punctuation, and collapses spaces.
        """
        if not s:
            return ""
        v = s.lower()
        # Normalize common Arabic variations and spelling variants
        mappings = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه', 'ڤ': 'ف'
        }
        for a, b in mappings.items():
            v = v.replace(a, b)
        # Remove characters that are not Arabic letters or numbers or spaces
        v = re.sub(r'[^0-9\u0600-\u06FF\s]', '', v)
        v = re.sub(r'\s+', ' ', v).strip()
        return v

    def _normalize_pid(self, p_or_pid) -> Optional[str]:
        """Return a stable string product id from a product dict or a pid-like value.

        Handles variants commonly seen in the codebase and exports:
        - Document with `product_id` field
        - Document with `_id` as an ObjectId instance
        - Document with `_id` as a dict like {'$oid': '...'} (JSON export)
        - Raw pid strings or ObjectId-like reprs (e.g. "ObjectId('...')")
        """
        pid = None
        if isinstance(p_or_pid, dict):
            # prefer explicit product_id when available
            pid = p_or_pid.get('product_id') or p_or_pid.get('productId') or p_or_pid.get('id') or p_or_pid.get('_id')
        else:
            pid = p_or_pid
        if pid is None:
            return None

        # If _id was exported as a dict with $oid, extract it
        if isinstance(pid, dict):
            if '$oid' in pid:
                return pid.get('$oid')
            if 'oid' in pid:
                return pid.get('oid')
            # fallback to string representation
            try:
                return str(pid)
            except Exception:
                return None

        # If it's an ObjectId instance, return its hex string
        try:
            from bson.objectid import ObjectId
            if isinstance(pid, ObjectId):
                return str(pid)
        except Exception:
            # bson may not be present in some test contexts; ignore
            pass

        # Normalize common wrapper reprs like "ObjectId('...')"
        s = str(pid)
        m = re.match(r"ObjectId\([\'\"]?([0-9a-fA-F]{24})[\'\"]?\)", s)
        if m:
            return m.group(1)

        return s

    def _select_products_by_issue(self, products: List[Dict[str, Any]], query: str, brand: Optional[str] = None, category: Optional[str] = None):
        """Try to select products that explicitly mention the user's issue in Arabic description/title/brand.
        Returns (header, data) if a tailored result can be produced, otherwise None to fall back to the general listing.
        """
        if not products:
            return None
        # Deduplicate
        deduped = []
        seen = set()
        for p in products:
            key = (p.get('product_id') or str(p.get('_id')) or (p.get('name') or '').strip()).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)

        q_s = self._simplify_text(query or "")
        if not q_s:
            return None

        matched_desc = []
        matched_other = []
        for p in deduped:
            matched_field = None
            excerpt = ''
            desc_s = self._simplify_text(p.get('description_ar') or '')
            title_s = self._simplify_text(p.get('title_ar') or p.get('title') or p.get('name') or '')
            brand_s = self._simplify_text(str(p.get('brand') or ''))
            # Break the query into tokens and match tokens / token variants against description/title/brand
            tokens = [t for t in re.split(r'\s+', q_s) if t and len(t) > 1]
            matched = False
            for tok in tokens:
                if not tok:
                    continue
                variants = [tok, 'مت' + tok, tok + 'ة', 'ال' + tok]
                if desc_s and any(v in desc_s for v in variants):
                    matched_field = 'description_ar'
                    desc = p.get('description_ar') or ''
                    found_idx = -1
                    for v in variants:
                        found_idx = desc.lower().find(v)
                        if found_idx >= 0:
                            break
                    if found_idx >= 0:
                        start = max(0, found_idx - 30)
                        excerpt = desc[start:start + 120].strip()
                    else:
                        excerpt = (desc[:120] or '').strip()
                    matched_desc.append((p, matched_field, excerpt))
                    matched = True
                    break
            if matched:
                continue
            # Check title or brand for token matches
            for tok in tokens:
                if tok and (tok in title_s or tok in brand_s):
                    matched_field = 'title_or_brand'
                    excerpt = p.get('title_ar') or p.get('title') or p.get('brand') or ''
                    matched_other.append((p, matched_field, excerpt))
                    matched = True
                    break
            if not matched:
                matched_other.append((p, None, ''))

        if matched_desc:
            selected = matched_desc[:10]
            header = f"وجدت {len(matched_desc)} منتجًا يذكر المشكلة في الوصف:\n"
        elif deduped:
            selected = matched_other[:10]
            header = f"لم أجد منتجًا يذكر النص بالضبط في الوصف، لكنها قد تكون ذات صلة:\n"
        else:
            return None

        # Build summaries
        summaries = []
        names = []
        for entry in selected:
            p, matched_field, excerpt = entry
            # Prefer Arabic title
            name = p.get('title_ar') or p.get('name') or p.get('title') or '(بدون اسم)'
            currency = p.get('currency') or ''
            price_info = self._get_price_info(p)
            price_val = price_info.get('min_price')
            in_stock = self._is_in_stock(p)
            availability = "متوفر" if in_stock else "غير متوفر"
            price_str = f"{price_val:.1f}" if isinstance(price_val, (int, float)) else (str(price_val) if price_val is not None else 'N/A')
            snippet = f" — {excerpt}" if excerpt else ''
            names.append(f"{p.get('title_ar') or name} - {price_str} {currency} - {availability}{snippet}")
            summaries.append({
                'product_id': self._normalize_pid(p),
                'name': name,
                'price': price_val,
                'price_map': p.get('price_map'),
                'currency': currency,
                'in_stock': in_stock,
                'stock_quantity': p.get('stock_quantity', 0),
                'category': p.get('category'),
                'brand': p.get('brand'),
                'image_url': p.get('image_url') or (p.get('images') or [None])[0],
                'match_field': matched_field,
                'match_excerpt': excerpt
            })

        numbered = []
        for i, n in enumerate(names, start=1):
            numbered.append(f"{i}. {n}")
        header += "\n".join(numbered)
        header += "\n\nللاطلاع على تفاصيل منتج، اكتب رقم المنتج (مثال: '1' أو '1 تفاصيل' أو 'رقم 1'). سأقبل الرقم حتى لو كتبته مع كلمات أخرى."
        # Ensure each product entry has a stable product_id (fallback to _id when needed)
        returned_products = []
        for p_entry, *_ in selected:
            # If the summary payload lacks identifiers, try to resolve the original DB document
            if not p_entry.get('product_id') or not p_entry.get('_id'):
                try:
                    q2 = {'$or': []}
                    if p_entry.get('title_ar'):
                        q2['$or'].append({'title_ar': {'$regex': re.escape(p_entry.get('title_ar')), '$options': 'i'}})
                    if p_entry.get('title'):
                        q2['$or'].append({'title': {'$regex': re.escape(p_entry.get('title')), '$options': 'i'}})
                    if p_entry.get('name'):
                        q2['$or'].append({'name': {'$regex': re.escape(p_entry.get('name')), '$options': 'i'}})
                    if q2['$or']:
                        orig = mongo_service.products.find_one(q2)
                        if orig:
                            # copy identifier fields back into the summary entry
                            p_entry['_id'] = orig.get('_id')
                            p_entry['product_id'] = orig.get('product_id')
                except Exception:
                    pass

            # Final normalization: prefer explicit product_id, otherwise normalize _id
            if not p_entry.get('product_id') and p_entry.get('_id'):
                p_entry['product_id'] = self._normalize_pid(p_entry.get('_id'))
            returned_products.append(p_entry)

        data = {'products': returned_products, 'summaries': summaries}
        return (header, data)

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Try to extract a brand and/or category mentioned in the user's query by
        comparing against distinct brands and categories from the product collection.
        Uses simplified normalized forms and alias maps to avoid mismatches due to spelling variants.
        Returns a dict containing possible keys:
         - brand: single match or None
         - category: single match or None
         - brand_candidates: list of candidates
         - category_candidates: list of candidates
        """
        q_norm = self._simplify_text(query or "")
        found_brand = None
        found_category = None
        found_type = None
        brand_candidates: List[str] = []
        category_candidates: List[str] = []
        type_candidates: List[str] = []

        try:
            brands = [b for b in mongo_service.products.distinct("brand") if b]
            categories = [c for c in mongo_service.products.distinct("category") if c]
            # Also collect product types (from attributes.type or top-level 'type') to support English type matches
            types = [t for t in mongo_service.products.distinct("attributes.type") if t] + [t for t in mongo_service.products.distinct('type') if t]
        except Exception:
            brands = []
            categories = []
            types = []

        # Build normalized lookup and prefer longest matches to avoid accidental short matches
        brands_norm = [(b, self._simplify_text(b)) for b in brands]
        categories_norm = [(c, self._simplify_text(c)) for c in categories]
        types_norm = [(t, self._simplify_text(t)) for t in types]

        # Check aliases first (user-friendly replacements)
        for alias, targets in self.category_aliases.items():
            if alias in q_norm:
                for t in targets:
                    if t in categories:
                        category_candidates.append(t)
                    elif t in types or any(self._simplify_text(t) == norm for _, norm in types_norm):
                        # detected a product 'type' (e.g., oil, shampoo) -> add to type candidates
                        type_candidates.append(t)
                    else:
                        # keep alias target as candidate so it can be shown to user as suggestion
                        category_candidates.append(t)
                # even if no exact target in categories/types, keep alias's targets as suggestions
                if not category_candidates and not type_candidates:
                    category_candidates.extend(targets)

        for alias, targets in self.brand_aliases.items():
            if alias in q_norm:
                for t in targets:
                    if t in brands:
                        brand_candidates.append(t)
                if not brand_candidates:
                    brand_candidates.extend(targets)

        # First pass: direct normalized substring match
        for original, norm in sorted(brands_norm, key=lambda x: -len(x[1])):
            if norm and norm in q_norm:
                found_brand = original
                break

        for original, norm in sorted(categories_norm, key=lambda x: -len(x[1])):
            if norm and norm in q_norm:
                found_category = original
                break

        # Second pass: token overlap fallback (helps when users omit parts or spell differently)
        if not found_brand:
            q_tokens = set(q_norm.split())
            for original, norm in sorted(brands_norm, key=lambda x: -len(x[1])):
                tokens = [t for t in norm.split() if len(t) > 1]
                if any(t in q_tokens for t in tokens):
                    found_brand = original
                    break

        if not found_category:
            q_tokens = set(q_norm.split())
            for original, norm in sorted(categories_norm, key=lambda x: -len(x[1])):
                tokens = [t for t in norm.split() if len(t) > 1]
                if any(t in q_tokens for t in tokens):
                    found_category = original
                    break

        # Also try to match product types (e.g., 'oil', 'shampoo') against query tokens
        if not found_type:
            q_tokens = set(q_norm.split())
            for original, norm in sorted(types_norm, key=lambda x: -len(x[1])):
                tokens = [t for t in norm.split() if len(t) > 1]
                if any(t in q_tokens for t in tokens):
                    found_type = original
                    break

        # If we have alias-based candidates but no exact found, keep candidates
        if not found_category and category_candidates:
            found_category = category_candidates[0]

        if not found_brand and brand_candidates:
            found_brand = brand_candidates[0]

        if not found_type and type_candidates:
            found_type = type_candidates[0]

        return {
            "brand": found_brand,
            "category": found_category,
            "type": found_type,
            "brand_candidates": brand_candidates,
            "category_candidates": category_candidates,
            "type_candidates": type_candidates
        }

    def _handle_search_intent(self, query: str):
        """Perform product search using possible filters extracted from the query.
        Returns (response_text, data)
        """
        # Normalize query for dialect/variants and early-detect generic browse requests
        q_norm = self.normalizer.normalize(query or "")
        # If user asked a generic browse question (e.g., 'ماهي المنتجات المتوفرة', 'شو هي المنتجات المتوفرة', 'اعرض الفئات'),
        # return grouped categories directly before doing a broad search
        if ((re.search(r'\b(ما|ماذا|ماهي|ما هي|عرض|اظهر|اعرض|شو|شو هي|شو في)\b', q_norm) and re.search(r'\b(المنتجات|منتجات|الفئات|فئات)\b', q_norm))
                or not q_norm.strip()):
            categories = [c for c in mongo_service.products.distinct('category') if c]
            grouped = []
            for c in categories:
                try:
                    cnt = mongo_service.products.count_documents({'category': c})
                    prods = mongo_service.search_products(query=None, category=c, limit=10)
                except Exception:
                    cnt = 0
                    prods = []

                # Get Arabic label
                if prods and len(prods) > 0:
                    cat_label = prods[0].get('category_ar') or prods[0].get('category') or c
                else:
                    try:
                        doc = mongo_service.products.find_one({'category': c}, {'category_ar': 1})
                        cat_label = doc.get('category_ar') if doc and doc.get('category_ar') else c
                    except Exception:
                        cat_label = c

                summaries_cat = []
                for p in prods:
                    name = p.get('title_ar') or p.get('name') or p.get('title') or "(بدون اسم)"
                    # Use normalized price info helper so we don't accidentally display raw dicts
                    pinfo = self._get_price_info(p)
                    price_val = pinfo.get('min_price')
                    in_stock = self._is_in_stock(p)
                    image = p.get('image_url') or (p.get('images') or p.get('image') or [None])[0]

                    summaries_cat.append({
                        "product_id": self._normalize_pid(p),
                        "name": name,
                        "price": price_val,
                        "in_stock": in_stock,
                        "image_url": image
                    })

                grouped.append({"category": cat_label, "count": cnt, "products": summaries_cat})

            msg = "هذه هي المنتجات المتوفرة مقسمة حسب الفئات. اكتب اسم فئة لعرض منتجاتها أو اكتب كلمة بحث أخرى."
            data = {"categories": grouped}
            return (msg, data)

        filters = self._extract_filters_from_query(query)
        brand = filters.get("brand")
        category = filters.get("category")
        prod_type = filters.get('type')

        # If we have explicit brand/category/type filters, prefer filtered search (regex matching)
        # Detect generic category/type browse requests (used later to skip issue-aware prioritization)
        is_generic_category_request = False
        try:
            if brand or category or prod_type:
                # Detect generic type/category browse queries (e.g., 'اريد زيت للشعر') and treat them
                # as a request to list the category/type products rather than searching for symptom-specific matches.
                q_norm_short = (query or '').strip().lower()
                # remove explicit filter words (brand/category/type) and common browse verbs
                if q_norm_short:
                    for val in (brand, category, prod_type):
                        if val:
                            try:
                                q_norm_short = re.sub(re.escape(val.lower()), '', q_norm_short)
                            except Exception:
                                pass
                    q_norm_short = re.sub(r'\b(اريد|أريد|ابغى|أبغى|عايز|احتاج|اعاني|اظهر|اعرض|عرض|أظهر|ما|ماذا|ماهي|ما هي|من|ل|الى|لل|لـ)\b', '', q_norm_short)
                    q_norm_short = re.sub(r'[^\u0600-\u06FF\s]', '', q_norm_short)
                is_generic_category_request = (not q_norm_short or len(q_norm_short.split()) <= 1)
                # If the query mentions a product-type keyword (Arabic) and a category was detected,
                # treat it as a generic category/type browse (e.g., 'اريد زيت للشعر').
                arabic_type_keywords = ['زيت', 'شامبو', 'كريم', 'سيروم', 'بلسم', 'قناع', 'سبراي', 'جل', 'لوشن', 'مصل']
                if category and any(k in (query or '') for k in arabic_type_keywords):
                    is_generic_category_request = True

                # If generic, list category/type products directly
                if is_generic_category_request:
                    # For generic browse requests, list the category's products (don't over-filter by type)
                    products = mongo_service.search_products(query=None, category=category, limit=10)
                else:
                    # Prefer query-based search when a query is present to honor specific product mentions
                    products = None
                    if query:
                        products = mongo_service.search_products(query=query, category=category, brand=brand, product_type=prod_type, limit=10)
                    # If query-based search returned nothing, fall back to category/brand filters
                    if not products:
                        products = mongo_service.search_products(query=None, category=category, brand=brand, product_type=prod_type, limit=10)
            else:
                products = mongo_service.search_products(query=query, limit=10)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return ("عذراً، حدث خطأ أثناء البحث عن المنتجات.", None)

        if products and len(products) > 0:
            # Try a symptom/issue-aware selection first (prefer products that mention the issue in the Arabic description/title/brand)
            # Skip this prioritization when the request was detected as a generic category/type browse
            try:
                if not is_generic_category_request:
                    smart = self._select_products_by_issue(products, query, brand=brand, category=category)
                    if smart:
                        return smart
            except Exception:
                # If the helper fails for any reason, fall back to the generic listing below
                pass

            # Build readable summaries with price and availability
            summaries = []
            names = []
            for p in products[:10]:
                # Prefer Arabic title when user query looks Arabic
                if re.search(r'[0-\u06FF]', str(query or '')) and p.get('title_ar'):
                    name = p.get('title_ar')
                else:
                    name = p.get("name") or p.get("title") or "(بدون اسم)"
                # Determine display price and availability using helpers
                currency = p.get("currency") or ""
                price_info = self._get_price_info(p)
                price_val = price_info.get('min_price')
                in_stock = self._is_in_stock(p)
                availability = "متوفر" if in_stock else "غير متوفر"
                # Format price with one decimal for readability (e.g., 15.0 SAR)
                if isinstance(price_val, (int, float)):
                    price_str = f"{price_val:.1f}"
                else:
                    price_str = str(price_val) if price_val is not None else 'N/A'
                names.append(f"{p.get('title_ar') or name} - {price_str} {currency} - {availability}")

                summaries.append({
                    "product_id": self._normalize_pid(p),
                    "name": name,
                    "price": price_val,
                    "price_map": p.get('price_map'),
                    "currency": currency,
                    "in_stock": in_stock,
                    "stock_quantity": p.get("stock_quantity", 0),
                    "category": p.get("category"),
                    "brand": p.get("brand"),
                    "image_url": p.get("image_url") or (p.get('images') or [None])[0]
                })

            # Build a numbered list and instruct the user how to select a product
            header = f"وجدت {len(products)} منتجًا"
            if brand:
                header += f" من {brand}"
            if category:
                header += f" في فئة {category}"
            header += ":\n"
            numbered = []
            for i, n in enumerate(names, start=1):
                numbered.append(f"{i}. {n}")
            header += "\n".join(numbered)
            header += "\n\nللاطلاع على تفاصيل منتج، اكتب رقم المنتج (مثال: '1' أو '1 تفاصيل' أو 'رقم 1'). سأقبل الرقم حتى لو كتبته مع كلمات أخرى."
            data = {"products": products, "summaries": summaries}
            return (header, data)
        else:
            # No products found - compute clarifying suggestions
            try:
                categories = [c for c in mongo_service.products.distinct('category') if c]
                brands = [b for b in mongo_service.products.distinct('brand') if b]
                q_norm = self._simplify_text(query)
                import difflib
                cat_matches = difflib.get_close_matches(q_norm, [self._simplify_text(c) for c in categories], n=3, cutoff=0.4)
                brand_matches = difflib.get_close_matches(q_norm, [self._simplify_text(b) for b in brands], n=3, cutoff=0.4)

                # map back to originals
                cat_suggestions = [categories[[self._simplify_text(c) for c in categories].index(m)] for m in cat_matches] if cat_matches else []
                brand_suggestions = [brands[[self._simplify_text(b) for b in brands].index(m)] for m in brand_matches] if brand_matches else []

                # include alias candidates if present
                alias_cats = filters.get('category_candidates') or []
                alias_brands = filters.get('brand_candidates') or []

                # Merge suggestions
                for a in alias_cats:
                    if a not in cat_suggestions:
                        cat_suggestions.append(a)
                for a in alias_brands:
                    if a not in brand_suggestions:
                        brand_suggestions.append(a)

                # If the user asked generically about available products (e.g. "ماهي المنتجات المتوفرة")
                # or provided an empty/very short query, return all categories with sample products grouped by category.
                browse_q = query or ""
                if ((re.search(r'\b(ما|ماذا|ماهي|ما هي|عرض|اظهر|اعرض)\b', browse_q) and re.search(r'\b(المنتجات|منتجات|الفئات|فئات)\b', browse_q))
                        or not browse_q.strip()):
                    grouped = []
                    for c in categories:
                        try:
                            cnt = mongo_service.products.count_documents({'category': c})
                            # fetch a small sample of products for each category
                            prods = mongo_service.search_products(query=None, category=c, limit=10)
                        except Exception:
                            cnt = 0
                            prods = []

                        # Determine Arabic label for the category - prefer explicit category_ar from any doc in DB
                        try:
                            doc = mongo_service.products.find_one({'category': c}, {'category_ar': 1})
                            cat_label = doc.get('category_ar') if doc and doc.get('category_ar') else (prods[0].get('category_ar') if prods and len(prods)>0 else c)
                        except Exception:
                            cat_label = (prods[0].get('category_ar') if prods and len(prods)>0 else c)

                        summaries_cat = []
                        for p in prods:
                            # Prefer Arabic title when available
                            name = p.get('title_ar') or p.get('name') or p.get('title') or "(بدون اسم)"
                            # Determine display price: prefer price_map min, then min_price, then price
                            price_val = None
                            pm = p.get('price_map') or {}
                            if isinstance(pm, dict) and pm:
                                try:
                                    price_val = min([float(v) for v in pm.values()])
                                except Exception:
                                    price_val = None
                            if price_val is None:
                                price_val = p.get('min_price') if p.get('min_price') is not None else p.get('price')
                            in_stock = p.get('in_stock', p.get('inStock', False))
                            image = p.get('image_url') or (p.get('images') or p.get('image') or [None])[0]

                            summaries_cat.append({
                                "product_id": self._normalize_pid(p),
                                "name": name,
                                "price": price_val,
                                "in_stock": in_stock,
                                "image_url": image
                            })

                        grouped.append({"category": cat_label, "count": cnt, "products": summaries_cat})

                    if grouped:
                        msg = "هذه هي المنتجات المتوفرة مقسمة حسب الفئات. اكتب اسم فئة لعرض منتجاتها أو اكتب كلمة بحث أخرى."
                        data = {"categories": grouped}
                        return (msg, data)

                if cat_suggestions or brand_suggestions:
                    msg = "لم أجد منتجات تطابق طلبك تمامًا. هل تقصد واحدة من هذه الفئات أو الماركات؟\n"
                    if cat_suggestions:
                        msg += "فئات مشابهة:\n• " + "\n• ".join(cat_suggestions) + "\n"
                    if brand_suggestions:
                        msg += "ماركات مشابهة:\n• " + "\n• ".join(brand_suggestions) + "\n"
                    msg += "أو هل تريد أن أجرب كلمات بحث أخرى؟"
                    data = {"clarify_options": {"categories": cat_suggestions, "brands": brand_suggestions}}
                    return (msg, data)
            except Exception as e:
                logger.debug(f"Clarify suggestion failed: {e}")

            return ("لم أجد منتجات تطابق طلبك. هل تريد أن أجرب كلمات بحث أخرى؟", None)

    def _parse_price_range(self, query: str) -> Dict[str, Optional[float]]:
        """Parse price range or single price expressions from Arabic or English text.
        Returns a dict: {"min": float|None, "max": float|None, "cheapest": bool}
        """
        s = query or ""
        # map Arabic-Indic digits to western
        trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        s_norm = s.translate(trans)

        cheapest = False
        most_expensive = False
        if 'ارخص' in s_norm or 'الأرخص' in s_norm:
            cheapest = True
        if 'اغلى' in s_norm or 'اعلى' in s_norm or 'الأغلى' in s_norm:
            most_expensive = True

        # patterns like 'بين 10 و 50' or 'بين ال 10 وال 50' or 'من 10 الى 50'
        m = re.search(r'بين\s*ال?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:و|و?ال)\s*([0-9]+(?:\.[0-9]+)?)', s_norm)
        if not m:
            m = re.search(r'من\s*([0-9]+(?:\.[0-9]+)?)\s*(?:إلى|الى|وبين|و)\s*([0-9]+(?:\.[0-9]+)?)', s_norm)
        if m:
            try:
                a = float(m.group(1))
                b = float(m.group(2))
                mn, mx = min(a, b), max(a, b)
                return {"min": mn, "max": mx, "cheapest": cheapest}
            except Exception:
                pass

        # single number (e.g., 'اقل من 50' or 'اقل من 50 ريال')
        m2 = re.search(r'اقل\s*من\s*([0-9]+(?:\.[0-9]+)?)', s_norm)
        if m2:
            try:
                return {"min": None, "max": float(m2.group(1)), "cheapest": cheapest}
            except Exception:
                pass

        m3 = re.search(r'اكبر\s*من\s*([0-9]+(?:\.[0-9]+)?)', s_norm)
        if m3:
            try:
                return {"min": float(m3.group(1)), "max": None, "cheapest": cheapest}
            except Exception:
                pass

        # fallback: any two numbers in the query
        nums = re.findall(r'([0-9]+(?:\.[0-9]+)?)', s_norm)
        if len(nums) >= 2:
            try:
                a, b = float(nums[0]), float(nums[1])
                mn, mx = min(a, b), max(a, b)
                return {"min": mn, "max": mx, "cheapest": cheapest}
            except Exception:
                pass

        return {"min": None, "max": None, "cheapest": cheapest, "most_expensive": most_expensive}

    def _handle_price_intent(self, query: str):
        """Handle price-based queries: cheapest, price ranges, etc."""
        parsed = self._parse_price_range(query)
        price_min = parsed.get('min')
        price_max = parsed.get('max')
        cheapest = parsed.get('cheapest')
        most_expensive = parsed.get('most_expensive') or False

        filters = self._extract_filters_from_query(query)
        brand = filters.get('brand')
        category = filters.get('category')

        q_norm = self._simplify_text(query or "")
        generic_product_request = ('منتج' in q_norm or 'منتجات' in q_norm)

        # Execute the price search: handle cheapest vs range and respect brand/category filters
        try:
            logger.debug("Price search parsed=%s filters=%s", parsed, {'brand': brand, 'category': category})
            # Determine sort direction and limit based on cheapest / most_expensive
            if cheapest:
                search_q = None if (not brand and not category and generic_product_request) else (query if not (brand or category) else None)
                logger.debug("Price search query used: %r (cheapest)", search_q)
                # fetch a small batch sorted by price ascending in case of ties and pick preferred
                batch = mongo_service.search_products(query=search_q, category=category, brand=brand, limit=10,
                                                     sort_by='price', sort_order=1)
                if batch:
                    # find the minimum price in batch
                    min_price_val = min([p.get('price') or 0 for p in batch])
                    tied = [p for p in batch if (p.get('price') or 0) == min_price_val]
                    if len(tied) > 1:
                        chosen = self._choose_preferred_product(tied)
                        products = [chosen] if chosen else tied[:1]
                    else:
                        products = tied
                else:
                    products = []
            elif most_expensive:
                search_q = None if (not brand and not category and generic_product_request) else (query if not (brand or category) else None)
                logger.debug("Price search query used: %r (most_expensive)", search_q)
                batch = mongo_service.search_products(query=search_q, category=category, brand=brand, limit=10,
                                                     sort_by='price', sort_order=-1)
                if batch:
                    max_price_val = max([p.get('price') or 0 for p in batch])
                    tied = [p for p in batch if (p.get('price') or 0) == max_price_val]
                    if len(tied) > 1:
                        chosen = self._choose_preferred_product(tied)
                        products = [chosen] if chosen else tied[:1]
                    else:
                        products = tied
                else:
                    products = []
            else:
                search_q = None if (not brand and not category and generic_product_request) else (query if not (brand or category) else None)
                logger.debug("Price search query used: %r (range)", search_q)
                products = mongo_service.search_products(query=search_q, category=category, brand=brand, limit=10,
                                                        min_price=price_min, max_price=price_max,
                                                        sort_by='price', sort_order=1)
            logger.debug("Price search returned %d products", len(products))
        except Exception as e:
            logger.warning(f"Price search failed: {e}")
            return ("عذراً، حدث خطأ أثناء البحث عن المنتجات حسب السعر.", None)

        if products and len(products) > 0:
            summaries = []
            names = []
            for idx, p in enumerate(products[:10], start=1):
                name = p.get('title_ar') or p.get('name') or p.get('title') or "(بدون اسم)"
                currency = p.get("currency") or ""
                # Determine display price (use price_map min if present)
                price_val = None
                pm = p.get('price_map') or {}
                if isinstance(pm, dict) and pm:
                    try:
                        price_val = min([float(v) for v in pm.values()])
                    except Exception:
                        price_val = None
                if price_val is None:
                    price_val = p.get('min_price') if p.get('min_price') is not None else p.get('price')
                in_stock = p.get("in_stock", False)
                availability = "متوفر" if in_stock else "غير متوفر"
                names.append(f"{idx}. {name} - {price_val or 'N/A'} {currency} - {availability}")

                summaries.append({
                    "index": idx,
                    "product_id": p.get("product_id"),
                    "name": name,
                    "price": price_val,
                    "currency": currency,
                    "in_stock": in_stock,
                    "stock_quantity": p.get("stock_quantity", 0),
                    "category": p.get("category"),
                    "brand": p.get("brand"),
                    "image_url": p.get("image_url")
                })

            header = f"وجدت {len(products)} منتجًا حسب الشرط"
            header += ":\n• " + "\n• ".join(names)
            header += "\nهل تريد تفاصيل أحد هذه المنتجات؟"
            data = {"products": products, "summaries": summaries}
            return (header, data)
        else:
            # If no products, produce clarification similar to search
            msg, data = self._handle_search_intent(query)
            # _handle_search_intent will return clarifying suggestions when nothing is found
            return (msg, data)

    def _handle_best_intent(self, query: str):
        """Handle 'best' product requests (sort by rating). Returns top result(s).
        Supports general requests and scoped by category/brand."""
        filters = self._extract_filters_from_query(query)
        brand = filters.get('brand')
        category = filters.get('category')
        q_norm = self._simplify_text(query or "")
        generic_product_request = ('منتج' in q_norm or 'منتجات' in q_norm)

        # If user asked plural 'منتجات' or 'أفضل منتجات', return multiple results (5),
        # otherwise return the single best product
        limit = 5 if 'منتجات' in q_norm else 1
        search_q = None if (not brand and not category and generic_product_request) else (query if not (brand or category) else None)

        try:
            products = mongo_service.search_products(query=search_q, category=category, brand=brand, limit=limit,
                                                    sort_by='rating', sort_order=-1)
        except Exception as e:
            logger.warning(f"Best-product search failed: {e}")
            return ("عذراً، حدث خطأ أثناء البحث عن أفضل المنتجات.", None)

        if products and len(products) > 0:
            summaries = []
            names = []
            for idx, p in enumerate(products[:limit], start=1):
                name = p.get('title_ar') or p.get('name') or p.get('title') or "(بدون اسم)"
                currency = p.get("currency") or ""
                # Determine display price (use price_map min if present)
                price_val = None
                pm = p.get('price_map') or {}
                if isinstance(pm, dict) and pm:
                    try:
                        price_val = min([float(v) for v in pm.values()])
                    except Exception:
                        price_val = None
                if price_val is None:
                    price_val = p.get('min_price') if p.get('min_price') is not None else p.get('price')
                in_stock = p.get("in_stock", False)
                availability = "متوفر" if in_stock else "غير متوفر"
                names.append(f"{idx}. {name} - {price_val or 'N/A'} {currency} - {availability}")

                summaries.append({
                    "index": idx,
                    "product_id": p.get("product_id"),
                    "name": name,
                    "price": price_val,
                    "currency": currency,
                    "in_stock": in_stock,
                    "stock_quantity": p.get("stock_quantity", 0),
                    "category": p.get("category"),
                    "brand": p.get("brand"),
                    "image_url": p.get("image_url")
                })

            header = f"أفضل {limit} منتجات" if limit > 1 else "أفضل منتج"
            if brand:
                header += f" من {brand}"
            if category:
                header += f" في فئة {category}"
            header += ":\n• " + "\n• ".join(names)
            header += "\nهل تريد تفاصيل أحد هذه المنتجات؟"
            data = {"products": products, "summaries": summaries}
            return (header, data)
        else:
            return ("لم أجد منتجات مناسبة للمعايير المطلوبة.", None)

    def _parse_customer_info(self, text: str) -> Dict[str, Optional[str]]:
        """Try to extract a phone number and a name from free text. Returns {name, phone}"""
        s = text or ""
        # translate Arabic-Indic digits
        trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        s_norm = s.translate(trans)

        # find phone number (simple heuristic: 7+ digits)
        m = re.search(r'(\+?\d[\d\s\-]{6,}\d)', s_norm)
        phone = None
        name = None
        if m:
            phone = re.sub(r'[^0-9+]', '', m.group(1))
            # name is remaining text without phone
            name_candidate = s_norm.replace(m.group(1), '').strip(' ,|-:\n')
            if name_candidate:
                name = name_candidate
        else:
            # try to find Arabic-Indic digits sequence
            m2 = re.search(r'([0-9]{7,})', s_norm)
            if m2:
                phone = m2.group(1)
                name_candidate = s_norm.replace(m2.group(1), '').strip(' ,|-:\n')
                if name_candidate:
                    name = name_candidate

        # fallback: if text looks like 'Name, 050...' or 'Name 050...'
        if not phone and ',' in s_norm:
            parts = [p.strip() for p in s_norm.split(',') if p.strip()]
            for p in parts:
                if re.search(r'\d', p):
                    phone = re.sub(r'[^0-9+]', '', p)
                else:
                    name = p

        return {"name": (name or '').strip(), "phone": (phone or '').strip()}

    def _handle_buy_intent(self, query: str, user_id: str):
        """Handle buy/add-to-cart intent. Collects product selection and customer info, then persists cart."""
        # Try to identify product from query or from last search context
        filters = self._extract_filters_from_query(query)
        # If user referenced product explicitly, try to find it by name
        product = None
        if query and len(query.strip()) > 2:
            candidates = mongo_service.search_products(query=query, limit=3)
            if candidates:
                product = candidates[0]

        # If no product found, but user has recent search results, use first
        last = self.search_context.get(user_id)
        # Normalize last search/resolution context to a list of summaries or products
        last_list = []
        if isinstance(last, dict):
            last_list = last.get('summaries') or last.get('products') or []
        else:
            last_list = last or []
        if not product and last_list:
            first = last_list[0]
            # If first element is a summary with product_id, fetch full product; otherwise assume it's already a product
            if isinstance(first, dict):
                pid = first.get('product_id') or first.get('_id')
                if pid:
                    product = mongo_service.get_product_by_id(pid)
                else:
                    product = first
            else:
                product = first

        if not product:
            return ("عن أي منتج تريد الشراء؟ الرجاء ذكر اسم المنتج أو اختيار رقم المنتج من القائمة.", None)

        # If user provided name+phone in same message, parse and complete
        cust = self._parse_customer_info(query)
        phone = cust.get('phone')
        name = cust.get('name') or 'عميل'
        if phone:
            # If product has multiple sizes, require size selection before adding to cart
            pm = product.get('price_map') or {}
            try:
                if isinstance(pm, dict) and len(pm) > 1:
                    sizes = list(pm.keys())
                    choices = []
                    for i, s in enumerate(sizes):
                        val = pm.get(s)
                        price_str = f"{val:.1f}" if isinstance(val, (int, float)) else str(val or 'N/A')
                        choices.append(f"{i+1}. {s} - {price_str} {product.get('currency','') or ''}")
                    # Persist awaiting size selection
                    st = conv_memory.get_user_state(user_id) or {}
                    st['awaiting_size_selection'] = {'product': product, 'sizes': sizes}
                    st['pending_buy'] = {'product': product, 'awaiting': 'size_selection'}
                    st['selected_product_id'] = product.get('product_id') or product.get('_id')
                    conv_memory.set_user_state(user_id, st)
                    prompt = "المنتج يحتوي على أحجام متعددة، يرجى اختيار الرقم المقابل للحجم الذي تريد:\n" + "\n".join(choices)
                    return (prompt, None)
            except Exception:
                pass

            try:
                pid = product.get('product_id') or product.get('_id')
                # Record pending buy; do NOT create cart or add item server-side
                st = conv_memory.get_user_state(user_id) or {}
                st['pending_buy'] = {'product': product, 'awaiting': 'confirm_checkout', 'quantity': 1}
                st['selected_product_id'] = pid
                conv_memory.set_user_state(user_id, st)
                response = f"سُجلت رغبتك بشراء المنتج. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
                product_title = (product.get('title_ar') or product.get('title') or product.get('name')) if product else None
                data = {"action": "start_add_to_cart", "product_title": product_title, "selected_size": None, "purchase_intent": True, "status": "started", "ask_confirm": True}
                return (response, data)
            except Exception as e:
                logger.warning(f"Buy flow failed: {e}")
                return ("عذراً، لم أتمكن من إتمام عملية الشراء الآن.", None)

        # Otherwise, ask for customer info and set pending state
        state = self._get_state(user_id)
        pid = product.get('product_id') or product.get('_id')
        # determine default selected size if single-size
        pinfo = self._get_price_info(product)
        pm = pinfo.get('price_map') or {}
        chosen_size = None
        if isinstance(pm, dict) and len(pm) == 1:
            chosen_size = list(pm.keys())[0]

        state['pending_buy'] = {'product': product, 'awaiting': 'confirm_checkout', 'quantity': 1}
        state['selected_product_id'] = pid
        self._set_state(user_id, state)
        # Inform user that add-to-cart was started and ask them to confirm (no name/phone required)
        prompt = "تم بدء إضافة المنتج إلى سلة التسوق بنجاح. هل ترغب بتأكيد الطلب الآن وإتمام الشراء؟ اكتب 'نعم' للتأكيد أو 'لا' للإلغاء."
        try:
            product_title = (product.get('title_ar') or product.get('title') or product.get('name')) if product else None
            self._log_bot_turn(user_id, prompt, 'buy', {'product_title': product_title})
        except Exception:
            pass
        data = {
            "action": "start_add_to_cart",
            "product_title": product_title,
            "selected_size": chosen_size,
            "purchase_intent": True,
            "status": "started",
            "ask_confirm": True
        }
        return (prompt, data)

    def _choose_preferred_product(self, products: list):
        """From a list of products (same price), prefer:
        1) in-stock products over out-of-stock
        2) higher rating
        3) higher review_count
        4) fallback to the first product
        Returns a single product dict.
        """
        if not products:
            return None
        # Prefer in_stock
        in_stock = [p for p in products if p.get('in_stock')]
        candidates = in_stock if in_stock else products
        # Sort by rating desc then review_count desc
        def score(p):
            r = p.get('rating') or 0
            rc = p.get('review_count') or 0
            return (r, rc)
        best = sorted(candidates, key=lambda p: score(p), reverse=True)[0]
        return best


    def _handle_detail_request(self, user_id: str, query: str):
        """Resolve a detail request from user's last search context.
        Returns (response_text, data) where data contains the full product details.
        """
        last = self.search_context.get(user_id)
        if not last:
            return ("ليس لدي نتائج سابقة. هل تريد أن أبحث عن منتج لك؟", None)

        # Support both the old style (list of summaries) and new style (dict with summaries/products)
        summaries = last.get('summaries') if isinstance(last, dict) else last
        logger.info("Detail request summaries preview: %s", [ {k: s.get(k) for k in ['product_id','name','title','title_ar']} for s in (summaries or [])[:5] ])

        # Try to detect an index (1-based) in the user's message
        m = re.search(r'([0-9]+|[٠١٢٣٤٥٦٧٨٩]+)', query)
        selected = None
        if m:
            raw = m.group(1)
            # Convert Arabic-Indic digits to normal digits if necessary
            trans = str.maketrans('٠١٢٣٤٥٦٧٨-٩', '0123456789')
            try:
                idx = int(raw.translate(trans)) - 1
                if 0 <= idx < len(summaries):
                    selected = summaries[idx]
            except Exception:
                selected = None

        # If no numeric selection, try to match by name tokens
        if not selected:
            q_norm = self._simplify_text(query)
            for s in summaries:
                name_norm = self._simplify_text(s.get('name', '') or s.get('title', ''))
                if name_norm and (name_norm in q_norm or any(tok in q_norm for tok in name_norm.split() if len(tok) > 1)):
                    selected = s
                    break

        if not selected:
            # Could not determine which product; ask user to clarify with numbered choice
            choices = [f"{i+1}. {s.get('name', s.get('title', ''))}" for i, s in enumerate(summaries[:5])]
            prompt = "لم أتمكن من تحديد المنتج. أي واحد تقصد؟\n" + "\n".join(choices)
            return (prompt, None)

        # DEBUG: log selected summary/product preview
        logger.info("Detail selected summary: %s", {k: selected.get(k) for k in ['product_id', 'name', 'title', 'title_ar']})

        # Fetch full product details
        product = None
        # remember the product as last viewed so subsequent 'نعم' (yes) can start a buy flow
        # (we set this after fetching full product below)
        if selected.get('product_id'):
            product = mongo_service.get_product_by_id(selected['product_id'])
        if not product:
            # Fallback by name/title
            product = mongo_service.get_product_by_name(selected.get('name', '') or selected.get('title', ''))
            logger.info("Product fetched by name lookup: %s", {'title': product.get('title') if product else None, 'title_ar': product.get('title_ar') if product else None, 'product_id': product.get('product_id') if product else None, '_id': str(product.get('_id')) if product and product.get('_id') else None})

        if not product:
            return ("عذرًا، لم أتمكن من جلب تفاصيل المنتج الآن.", None)

        # Ensure a stable product_id exists even for legacy docs that only have _id
        if not product.get('product_id'):
            if product.get('_id'):
                product['product_id'] = self._normalize_pid(product.get('_id'))
            else:
                # finalize fallback to any identifier we can normalize
                product['product_id'] = self._normalize_pid(product) or None

        # Try to augment product with original DB fields (product_id/_id) when missing
        if not product.get('product_id') or not product.get('_id'):
            try:
                q = {'$or': []}
                if product.get('product_id'):
                    q['$or'].append({'product_id': product.get('product_id')})
                title_ar = product.get('title_ar') or ''
                title = product.get('title') or ''
                name = product.get('name') or ''
                desc = product.get('description') or product.get('description_ar') or ''
                if title_ar:
                    q['$or'].append({'title_ar': {'$regex': re.escape(title_ar), '$options': 'i'}})
                if title:
                    q['$or'].append({'title': {'$regex': re.escape(title), '$options': 'i'}})
                if name:
                    q['$or'].append({'name': {'$regex': re.escape(name), '$options': 'i'}})
                if desc:
                    # small snippet to avoid overlong regex
                    q['$or'].append({'description': {'$regex': re.escape(desc[:80]), '$options': 'i'}})
                if q['$or']:
                    logger.info("Augmentation query for product: %s product_preview=%s", q, {'title': product.get('title'), 'title_ar': product.get('title_ar')})
                    orig = mongo_service.products.find_one({'$or': q['$or']})
                    logger.info("Augmentation found orig: %s", {'_id': str(orig.get('_id')) if orig and orig.get('_id') else None, 'product_id': orig.get('product_id') if orig else None, 'title': orig.get('title') if orig else None, 'title_ar': orig.get('title_ar') if orig else None})
                    if not orig:
                        # Try permissive tokenized lookup by simplified title tokens
                        try:
                            title_tok = self._simplify_text(product.get('title_ar') or product.get('title') or '')
                            for tok in [t for t in title_tok.split() if len(t) > 3]:
                                logger.info("Trying permissive lookup token=%s", tok)
                                maybe = mongo_service.products.find_one({'$or': [{'title_ar': {'$regex': tok, '$options': 'i'}}, {'title': {'$regex': tok, '$options': 'i'}}]})
                                if maybe:
                                    orig = maybe
                                    logger.info("Permissive lookup found: %s", {'_id': str(orig.get('_id')) if orig and orig.get('_id') else None, 'product_id': orig.get('product_id'), 'title_ar': orig.get('title_ar')})
                                    break
                        except Exception as e:
                            logger.debug("Permissive lookup failed: %s", e)
                    if orig:
                        # copy missing identifiers
                        if not product.get('product_id') and orig.get('product_id'):
                            product['product_id'] = orig.get('product_id')
                        if not product.get('_id') and orig.get('_id'):
                            product['_id'] = orig.get('_id')
            except Exception as e:
                logger.debug("Augmentation lookup failed: %s", e)

        # Build detail message with support for price maps (sizes)
        name = product.get('title_ar') or product.get('title') or product.get('name')
        desc = product.get('description_ar') or product.get('description') or ''
        # Normalize price info so prices stored under 'price' or 'price_map' are handled consistently
        pinfo = self._get_price_info(product)
        price_map = pinfo.get('price_map') or {}
        min_price = pinfo.get('min_price')
        # derive a max price if variants present
        max_price = product.get('max_price') if product.get('max_price') is not None else None
        if (max_price is None) and isinstance(price_map, dict) and price_map:
            try:
                vals = [float(v.get('price') if isinstance(v, dict) else v) for v in price_map.values()]
                if vals:
                    max_price = max(vals)
            except Exception:
                max_price = None
        currency = product.get('currency') or ''
        in_stock = self._is_in_stock(product)
        qty = product.get('stock_quantity', 0)
        availability = 'متوفر' if in_stock else 'غير متوفر'

        # Price header
        details = f"تفاصيل المنتج: {name}\nالسعر: {min_price if min_price is not None else 'N/A'} {currency}"
        if max_price is not None and min_price is not None and max_price != min_price:
            details += f" - يبدأ من {min_price} وحتى {max_price} {currency}"
        details += f"\nالتوفر: {availability} (كمية: {qty})\n"
        if desc:
            details += f"\nالوصف:\n{desc}\n"
        if price_map:
            details += "\nأحجام/أسعار متاحة:\n"
            for size, pr in price_map.items():
                # support variant records where pr might be dict
                if isinstance(pr, dict):
                    pv = pr.get('price') or pr.get('value') or pr.get('amount')
                else:
                    pv = pr
                price_str = f"{pv:.1f}" if isinstance(pv, (int, float)) else str(pv or 'N/A')
                details += f"- {size}: {price_str} {currency}\n"
        if product.get('images'):
            details += f"\nصور إضافية متاحة ({len(product.get('images'))})\n"
        if product.get('attributes'):
            details += f"\nالمواصفات:\n"
            for k, v in product.get('attributes', {}).items():
                details += f"- {k}: {v}\n"

        # Save last viewed product for this user so a following 'نعم' will start the buy flow
        state = self._get_state(user_id)
        state['last_viewed_product'] = product
        # Ensure we persist a stable selected_product_id (fallback to _id when product_id is absent)
        sel_pid = self._normalize_pid(product)
        state['selected_product_id'] = sel_pid
        logger.debug("selected_product_id set to %s for user %s", sel_pid, user_id)
        self._set_state(user_id, state)

        # Append buy prompt and include product in returned data
        details += "\nهل تريد شراء هذا المنتج؟ اكتب 'نعم' للموافقة أو اكتب 'لا' للإلغاء."
        data = {"product": product, "ask_buy": True}
        try:
            self._log_bot_turn(user_id, details, 'detail', {'product_id': product.get('product_id')})
        except Exception:
            pass
        return (details, data)

    def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation for a user"""
        if user_id in self.memory:
            del self.memory[user_id]
            return True
        return False

# Create singleton instance
chatbot_service = ChatbotService()