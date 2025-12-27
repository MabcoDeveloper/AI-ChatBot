import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Access product data for brand/category extraction
from services.mongo_service import mongo_service

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
        """Normalize Arabic text"""
        if not text:
            return ""
        
        normalized = text
        for dialect, standard in self.dialect_mappings.items():
            normalized = normalized.replace(dialect, standard)
        
        return normalized.strip()

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
    
    def _contains_keyword(self, text: str, kw: str) -> bool:
        """Match keyword only when not embedded inside other Arabic letters or numbers.
        This avoids false positives such as matching 'كم' inside 'لديكم'."""
        pattern = rf'(^|[^\u0600-\u06FF0-9]){re.escape(kw)}($|[^\u0600-\u06FF0-9])'
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

        # Check for search intent
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.search_keywords):
            return {"intent": "search", "confidence": 0.90}        
        # Check for offers intent
        if any(self._contains_keyword(text_lower, keyword) for keyword in self.offer_keywords):
            return {"intent": "offers", "confidence": 0.90}
        
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
        # Per-user transient state (pending buy info, etc.)
        self.user_state: Dict[str, Dict[str, Any]] = {}
        # Aliases for categories and brands (map common user words to canonical names)
        # Values can be lists; we try them in order when searching
        self.category_aliases: Dict[str, List[str]] = {
            'زيوت': ['العناية بالشعر', 'العناية بالبشرة'],
            'مرطب': ['العناية بالبشرة'],
            'شامبو': ['العناية بالشعر'],
            'عطر': ['العطور'],
            'عطور': ['العطور'],
            'كريم': ['العناية بالبشرة']
        }
        self.brand_aliases: Dict[str, List[str]] = {
            'نيفيا': ['نيڤيا'],
            'ماك': ['ماك'],
            'ذا أورديناري': ['ذا أورديناري']
        }
        
    def process_message(self, user_id: str, message: str, **kwargs):
        """
        Process user message - FIXED: removed session_data parameter
        Accepts **kwargs to handle extra parameters from UI
        """
        logger.info(f"Processing message from user {user_id}: {message}")
        
        # Normalize the message
        normalized = self.normalizer.normalize(message)
        
        # Detect intent
        intent_result = self.intent_detector.detect(normalized)
        intent = intent_result["intent"]
        
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

        # If we have a pending buy awaiting customer info, try to parse and complete the order
        state = self.user_state.get(user_id, {})
        pending_buy = state.get('pending_buy') if state else None
        if pending_buy and pending_buy.get('awaiting') == 'customer_info':
            cust = self._parse_customer_info(normalized)
            if cust.get('phone'):
                try:
                    cart = mongo_service.get_or_create_cart_by_customer(cust.get('name') or 'زائر', cust.get('phone'))
                    added = mongo_service.add_item_to_cart(cart['cart_id'], pending_buy['product']['product_id'], pending_buy.get('quantity', 1))
                    # clear pending state
                    state.pop('pending_buy', None)
                    self.user_state[user_id] = state
                    # Save cart ref in memory
                    self.memory[user_id].append({
                        "role": "bot",
                        "message": f"تم إضافة {pending_buy['product']['name']} إلى سلة التسوق باسم {cust.get('name')} ورقم {cust.get('phone')}.",
                        "intent": 'buy',
                        "timestamp": datetime.now().isoformat()
                    })
                    response_text = f"تم إضافة {pending_buy['product']['name']} إلى سلة التسوق باسم {cust.get('name')} ورقم {cust.get('phone')}. المجموع الحالي: {added.get('subtotal','N/A')} {added.get('currency','')}"
                    data = {"cart": added, "set_cookie": {"name": "cart_id", "value": added['cart_id'], "max_age": 30*24*3600}}
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
                    return ("عذراً، لم أتمكن من إضافة المنتج إلى السلة الآن.", None)
            # else: keep waiting for customer info

        # If the user replies with a pure numeric choice and we have recent search context,
        # treat it as selecting one of the presented products (e.g., '1', '٢', '3')
        if re.match(r'^\s*[0-9٠١٢٣٤٥٦٧٨٩]+[\.)]?\s*$', normalized) and self.search_context.get(user_id):
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
            state = self.user_state.get(user_id, {})
            last_viewed = state.get('last_viewed_product') if state else None
            if last_viewed and not state.get('pending_buy'):
                # start pending buy flow and ask for name/phone
                state['pending_buy'] = {'product': last_viewed, 'awaiting': 'customer_info', 'quantity': 1}
                self.user_state[user_id] = state
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
                last_summaries = self.search_context.get(user_id, [])
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
                    choices = [f"{i+1}. {s['name']}" for i, s in enumerate(last_summaries[:5])]
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
            except Exception:
                # fallback behavior
                if 'منتج' in simple or 'منتجات' in simple:
                    intent = 'search'
                    intent_result['confidence'] = 0.80

        # Handle search intent specially to return product results when possible
        if intent == 'search':
            response_text, data = self._handle_search_intent(normalized)
            # Keep last search summaries so follow-ups like 'تفاصيل ...' can refer to them
            if data and data.get('summaries'):
                self.search_context[user_id] = data.get('summaries')
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
        brand_candidates: List[str] = []
        category_candidates: List[str] = []

        try:
            brands = [b for b in mongo_service.products.distinct("brand") if b]
            categories = [c for c in mongo_service.products.distinct("category") if c]
        except Exception:
            brands = []
            categories = []

        # Build normalized lookup and prefer longest matches to avoid accidental short matches
        brands_norm = [(b, self._simplify_text(b)) for b in brands]
        categories_norm = [(c, self._simplify_text(c)) for c in categories]

        # Check aliases first (user-friendly replacements)
        for alias, targets in self.category_aliases.items():
            if alias in q_norm:
                for t in targets:
                    if t in categories:
                        category_candidates.append(t)
                # even if no exact target in categories, keep alias's targets as suggestions
                if not category_candidates:
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

        # If we have alias-based candidates but no exact found, keep candidates
        if not found_category and category_candidates:
            found_category = category_candidates[0]

        if not found_brand and brand_candidates:
            found_brand = brand_candidates[0]

        return {
            "brand": found_brand,
            "category": found_category,
            "brand_candidates": brand_candidates,
            "category_candidates": category_candidates
        }

    def _handle_search_intent(self, query: str):
        """Perform product search using possible filters extracted from the query.
        Returns (response_text, data)
        """
        filters = self._extract_filters_from_query(query)
        brand = filters.get("brand")
        category = filters.get("category")

        # If we have explicit brand/category filters, prefer filtered search (regex matching)
        try:
            if brand or category:
                products = mongo_service.search_products(query=None, category=category, brand=brand, limit=10)
                # If filters returned nothing, try a broader text search using the full query
                if (not products or len(products) == 0) and query:
                    products = mongo_service.search_products(query=query, category=category, brand=brand, limit=10)
            else:
                products = mongo_service.search_products(query=query, limit=10)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return ("عذراً، حدث خطأ أثناء البحث عن المنتجات.", None)

        if products and len(products) > 0:
            # Build readable summaries with price and availability
            summaries = []
            names = []
            for p in products[:10]:
                name = p.get("name", "(بدون اسم)")
                price = p.get("price")
                currency = p.get("currency") or ""
                in_stock = p.get("in_stock", False)
                availability = "متوفر" if in_stock else "غير متوفر"
                names.append(f"{name} - {price or 'N/A'} {currency} - {availability}")

                summaries.append({
                    "product_id": p.get("product_id"),
                    "name": name,
                    "price": p.get("price"),
                    "currency": currency,
                    "in_stock": in_stock,
                    "stock_quantity": p.get("stock_quantity", 0),
                    "category": p.get("category"),
                    "brand": p.get("brand"),
                    "image_url": p.get("image_url")
                })

            header = f"وجدت {len(products)} منتجًا"
            if brand:
                header += f" من {brand}"
            if category:
                header += f" في فئة {category}"
            header += ":\n• " + "\n• ".join(names)
            header += "\nهل تريد تفاصيل أحد هذه المنتجات؟"
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
            for p in products[:10]:
                name = p.get("name", "(بدون اسم)")
                price = p.get("price")
                currency = p.get("currency") or ""
                in_stock = p.get("in_stock", False)
                availability = "متوفر" if in_stock else "غير متوفر"
                names.append(f"{name} - {price or 'N/A'} {currency} - {availability}")

                summaries.append({
                    "product_id": p.get("product_id"),
                    "name": name,
                    "price": p.get("price"),
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
            for p in products[:limit]:
                name = p.get("name", "(بدون اسم)")
                price = p.get("price")
                currency = p.get("currency") or ""
                in_stock = p.get("in_stock", False)
                availability = "متوفر" if in_stock else "غير متوفر"
                names.append(f"{name} - {price or 'N/A'} {currency} - {availability}")

                summaries.append({
                    "product_id": p.get("product_id"),
                    "name": name,
                    "price": p.get("price"),
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
        if not product and last and len(last) > 0:
            product = last[0]

        if not product:
            return ("عن أي منتج تريد الشراء؟ الرجاء ذكر اسم المنتج أو اختيار رقم المنتج من القائمة.", None)

        # If user provided name+phone in same message, parse and complete
        cust = self._parse_customer_info(query)
        phone = cust.get('phone')
        name = cust.get('name') or 'عميل'
        if phone:
            try:
                cart = mongo_service.get_or_create_cart_by_customer(name, phone)
                updated = mongo_service.add_item_to_cart(cart['cart_id'], product.get('product_id'), 1)
                data = {"cart": updated, "set_cookie": {"name": "cart_id", "value": updated['cart_id'], "max_age": 30*24*3600}}
                response = f"تم إضافة {product.get('name')} إلى سلة التسوق باسم {name} ورقم {phone}. المجموع الحالي: {updated.get('subtotal','N/A')} {updated.get('currency','')}"
                return (response, data)
            except Exception as e:
                logger.warning(f"Buy flow failed: {e}")
                return ("عذراً، لم أتمكن من إتمام عملية الشراء الآن.", None)

        # Otherwise, ask for customer info and set pending state
        state = self.user_state.get(user_id, {})
        state['pending_buy'] = {'product': product, 'awaiting': 'customer_info', 'quantity': 1}
        self.user_state[user_id] = state
        prompt = "لتأكيد الشراء، يرجى تزويدي باسمك الكامل ورقم هاتفك (مثال: أحمد, 0501234567)"
        return (prompt, None)
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

        # Try to detect an index (1-based) in the user's message
        m = re.search(r'([0-9]+|[٠١٢٣٤٥٦٧٨٩]+)', query)
        selected = None
        if m:
            raw = m.group(1)
            # Convert Arabic-Indic digits to normal digits if necessary
            trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            try:
                idx = int(raw.translate(trans)) - 1
                if 0 <= idx < len(last):
                    selected = last[idx]
            except Exception:
                selected = None

        # If no numeric selection, try to match by name tokens
        if not selected:
            q_norm = self._simplify_text(query)
            for s in last:
                name_norm = self._simplify_text(s.get('name', ''))
                if name_norm and (name_norm in q_norm or any(tok in q_norm for tok in name_norm.split() if len(tok) > 1)):
                    selected = s
                    break

        if not selected:
            # Could not determine which product; ask user to clarify with numbered choice
            choices = [f"{i+1}. {s['name']}" for i, s in enumerate(last[:5])]
            prompt = "لم أتمكن من تحديد المنتج. أي واحد تقصد؟\n" + "\n".join(choices)
            return (prompt, None)

        # Fetch full product details
        product = None
        # remember the product as last viewed so subsequent 'نعم' (yes) can start a buy flow
        # (we set this after fetching full product below)
        if selected.get('product_id'):
            product = mongo_service.get_product_by_id(selected['product_id'])
        if not product:
            # Fallback by name
            product = mongo_service.get_product_by_name(selected.get('name', ''))

        if not product:
            return ("عذرًا، لم أتمكن من جلب تفاصيل المنتج الآن.", None)

        # Build detail message
        name = product.get('name')
        desc = product.get('description') or ''
        price = product.get('price')
        original = product.get('original_price')
        currency = product.get('currency') or ''
        in_stock = product.get('in_stock', False)
        qty = product.get('stock_quantity', 0)
        availability = 'متوفر' if in_stock else 'غير متوفر'

        details = f"تفاصيل المنتج: {name}\nالسعر: {price or 'N/A'} {currency}"
        if original:
            details += f" (السعر الأصلي: {original} {currency})"
        details += f"\nالتوفر: {availability} (كمية: {qty})\n"
        if desc:
            details += f"\nالوصف:\n{desc}\n"
        if product.get('attributes'):
            details += f"\nالمواصفات:\n"
            for k, v in product.get('attributes', {}).items():
                details += f"- {k}: {v}\n"

        # Save last viewed product for this user so a following 'نعم' will start the buy flow
        state = self.user_state.get(user_id, {})
        state['last_viewed_product'] = product
        self.user_state[user_id] = state

        # Append buy prompt and include product in returned data
        details += "\nهل تريد شراء هذا المنتج؟ اكتب 'نعم' للموافقة أو اكتب 'لا' للإلغاء."
        data = {"product": product, "ask_buy": True}
        return (details, data)

    def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation for a user"""
        if user_id in self.memory:
            del self.memory[user_id]
            return True
        return False

# Create singleton instance
chatbot_service = ChatbotService()