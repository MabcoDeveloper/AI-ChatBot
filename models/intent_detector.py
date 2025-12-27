import re
from typing import Dict, Any, List, Tuple
from config import settings

class IntentDetector:
    """Deterministic intent classifier for Arabic e-commerce"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.search_patterns = self._compile_patterns(settings.SEARCH_PATTERNS)
        self.price_patterns = self._compile_patterns(settings.PRICE_PATTERNS)
        self.offer_patterns = self._compile_patterns(settings.OFFER_PATTERNS)
        self.help_patterns = self._compile_patterns(settings.HELP_PATTERNS)
        
        # Beauty product categories (Arabic)
        self.beauty_categories = [
            "مكياج", "العناية بالبشرة", "العناية بالشعر", "العطور",
            "العناية بالجسم", "العناية بالأظافر", "مستحضرات تجميل",
            "كريم", "سيروم", "تونر", "ماسك", "شامبو", "بلسم",
            "أحمر شفاه", "أساس", "كونسيلر", "بودرة", "ملمع شفاه",
            "ماسكرا", "آيلاينر", "ظلال عيون", "بلاشر", "برونزر",
            "عطر", "مزيل عرق", "صابون", "لوشن", "زيت", "مقشر"
        ]
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """Compile regex patterns"""
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect(self, text: str) -> Dict[str, Any]:
        """Detect intent from normalized Arabic text"""
        
        # First, check for help intent (highest priority for confused users)
        if any(pattern.search(text) for pattern in self.help_patterns):
            return {
                "intent": "general_help",
                "confidence": 0.90,
                "parameters": {"help_type": "general"},
                "raw_text": text
            }
        
        # Check for offers intent
        if any(pattern.search(text) for pattern in self.offer_patterns):
            return {
                "intent": "offers",
                "confidence": 0.88,
                "parameters": {},
                "raw_text": text
            }
        
        # Check for price intent
        price_match = self._check_price_intent(text)
        if price_match[0]:
            return {
                "intent": "price_inquiry",
                "confidence": price_match[1],
                "parameters": {"product_name": price_match[2]},
                "raw_text": text
            }
        
        # Check for search intent
        search_match = self._check_search_intent(text)
        if search_match[0]:
            return {
                "intent": "search_product",
                "confidence": search_match[1],
                "parameters": {"query": search_match[2]},
                "raw_text": text
            }
        
        # Fallback intent - try to guess based on content
        fallback_intent = self._guess_fallback_intent(text)
        return fallback_intent
    
    def _check_search_intent(self, text: str) -> Tuple[bool, float, str]:
        """Check if text contains search intent"""
        # Check for search patterns
        has_pattern = any(pattern.search(text) for pattern in self.search_patterns)
        
        # Check for beauty-related keywords
        has_beauty_keyword = any(
            keyword in text.lower() for keyword in self.beauty_categories
        )
        
        if has_pattern or has_beauty_keyword:
            # Extract product query
            query = self._extract_product_query(text)
            confidence = 0.85 if has_pattern else 0.70
            return (True, confidence, query)
        
        # If text is a simple product name (1-3 words)
        words = text.split()
        if 1 <= len(words) <= 3 and any(w in self.beauty_categories for w in words):
            return (True, 0.65, text)
        
        return (False, 0.0, "")
    
    def _check_price_intent(self, text: str) -> Tuple[bool, float, str]:
        """Check if text contains price inquiry intent"""
        # Check for price patterns
        has_pattern = any(pattern.search(text) for pattern in self.price_patterns)
        
        if has_pattern:
            product_name = self._extract_product_name(text)
            confidence = 0.90 if product_name else 0.75
            return (True, confidence, product_name)
        
        # Check for "كم" followed by product
        if "كم" in text.lower():
            # Remove "كم" and check if rest looks like product
            without_km = text.lower().replace("كم", "").strip()
            words = without_km.split()
            if 1 <= len(words) <= 4:
                return (True, 0.70, without_km)
        
        return (False, 0.0, "")
    
    def _extract_product_query(self, text: str) -> str:
        """Extract product search query from text"""
        # Remove common question words and patterns
        remove_patterns = [
            r'أين (يوجد|هناك|يمكن إيجاد)?',
            r'وين (في|عندكم|عندك)?',
            r'ابحث عن',
            r'بدي',
            r'أريد',
            r'محتاج',
            r'عايز',
            r'هل (لديكم|يوجد|هناك)?',
            r'ما (هو|هي)?',
            r'شو (في|هو)?'
        ]
        
        query = text
        for pattern in remove_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # Remove question marks and extra spaces
        query = re.sub(r'[؟?\s]+', ' ', query).strip()
        
        # If query is empty after removal, use original text
        if not query:
            # Try to extract just the product/category words
            words = text.split()
            beauty_words = [w for w in words if w in self.beauty_categories]
            if beauty_words:
                query = " ".join(beauty_words)
            else:
                query = text
        
        return query[:100]  # Limit length
    
    def _extract_product_name(self, text: str) -> str:
        """Extract product name from price inquiry"""
        # Remove price-related words
        price_words = ["سعر", "كم", "بكم", "التكلفة", "الثمن", 
                      "السعر", "الكمية", "تكلفة", "قيمة", "ثمن"]
        
        product_name = text
        for word in price_words:
            # Remove word with optional space after
            pattern = r'\b' + re.escape(word) + r'\s*'
            product_name = re.sub(pattern, '', product_name, flags=re.IGNORECASE)
        
        # Remove question marks and common particles
        product_name = re.sub(r'[؟?\s]+', ' ', product_name).strip()
        
        # Remove common particles
        particles = ["هل", "ما", "من", "في", "على", "إلى", "عن", "مع"]
        words = product_name.split()
        filtered_words = [w for w in words if w not in particles]
        
        return " ".join(filtered_words).strip()
    
    def _guess_fallback_intent(self, text: str) -> Dict[str, Any]:
        """Guess intent for fallback cases"""
        words = text.split()
        
        # If text contains numbers, might be price inquiry
        if any(word.isdigit() for word in words):
            return {
                "intent": "price_inquiry",
                "confidence": 0.60,
                "parameters": {"product_name": text},
                "raw_text": text
            }
        
        # If text is short (1-3 words), might be search
        if 1 <= len(words) <= 3:
            return {
                "intent": "search_product",
                "confidence": 0.55,
                "parameters": {"query": text},
                "raw_text": text
            }
        
        # Default fallback
        return {
            "intent": "fallback",
            "confidence": 0.50,
            "parameters": {},
            "raw_text": text
        }

# Create singleton instance
intent_detector = IntentDetector()