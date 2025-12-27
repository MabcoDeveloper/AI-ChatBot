import re
from typing import Dict, Any
import pyarabic.araby as araby

class ArabicNormalizer:
    """Rule-based Arabic text normalizer with Syrian dialect support"""
    
    def __init__(self):
        # Syrian dialect to MSA mappings (expanded for beauty products)
        self.dialect_mappings = {
            # Questions
            "وين": "أين",
            "شو": "ما",
            "ليش": "لماذا",
            "إيمتا": "متى",
            "كيف": "كيف",
            "قديش": "كم",
            "شلون": "كيف",
            
            # Verbs and pronouns
            "بدي": "أريد",
            "بدنا": "نريد",
            "عم": "أعمل",
            "عم ب": "أ",
            "رح": "سوف",
            "حاول": "جرب",
            "شوف": "انظر",
            "قول": "قل",
            "روح": "اذهب",
            "جيب": "احضر",
            "عطيني": "أعطني",
            
            # Beauty product specific
            "مكياج": "مكياج",
            "ميكب": "مكياج",
            "شوارب": "شعر الوجه",
            "حواجب": "حواجب",
            "شفايف": "شفاه",
            
            # Common words
            "هون": "هنا",
            "هناك": "هناك",
            "هاي": "هذه",
            "هذاك": "ذلك",
            "كتير": "كثير",
            "منيح": "جيد",
            "طيب": "حسنا",
            "زلمة": "رجل",
            "ست": "امرأة",
            "بنت": "فتاة",
            "شغل": "عمل",
            "مش": "لا",
            "مو": "ليس",
            "ما": "لا",
            "والله": "حقا",
            "أكيد": "بالتأكيد",
            
            # Expressions
            "يلا": "هيا",
            "هيك": "هكذا",
            "طز": "لا يهم",
            "عنجد": "حقا",
            "ممتاز": "ممتاز",
            "حلو": "جميل",
            
            # Shopping terms
            "شراء": "شراء",
            "بيع": "بيع",
            "سوق": "سوق",
            "محل": "متجر",
            "مول": "مركز تسوق",
            "أونلاين": "عبر الإنترنت",
        }
        
        # Normalization patterns
        self.patterns = [
            (r'[ًٌٍَُِّْ]', ''),  # Remove diacritics
            (r'[إأآا]', 'ا'),    # Normalize alef
            (r'[ى]', 'ي'),       # Normalize ya
            (r'[ئءؤ]', 'ء'),     # Normalize hamza
            (r'[ة]', 'ه'),       # Normalize ta marbuta
            (r'\s+', ' '),       # Remove extra spaces
            (r'[،؛؟!\.\-_,]', ''),  # Remove punctuation
            (r'[^\w\sء-ي]', ''), # Remove non-Arabic characters except spaces
        ]
    
    def normalize(self, text: str) -> Dict[str, Any]:
        """Normalize Arabic text with Syrian dialect support"""
        if not text or not isinstance(text, str):
            return {"normalized": "", "original": text, "had_dialect": False}
        
        original = text.strip()
        normalized = original
        
        # Convert to lowercase for case-insensitive matching
        normalized_lower = normalized.lower()
        
        # Replace Syrian dialect words
        had_dialect = False
        for dialect, msa in self.dialect_mappings.items():
            # Match whole words
            pattern = r'\b' + re.escape(dialect) + r'\b'
            if re.search(pattern, normalized_lower, re.IGNORECASE):
                normalized = re.sub(
                    pattern, 
                    msa, 
                    normalized, 
                    flags=re.IGNORECASE
                )
                had_dialect = True
        
        # Apply normalization patterns
        for pattern, replacement in self.patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Normalize Arabic letters using pyarabic
        try:
            normalized = araby.normalize_hamza(normalized)
            normalized = araby.normalize_ligature(normalized)
            normalized = araby.normalize_alef(normalized)
        except:
            pass  # Fallback if pyarabic fails
        
        # Remove common filler words but keep important ones
        filler_words = ["و", "في", "على", "من", "الى", "عن", "مع", "هو", "هي", "هم"]
        words = normalized.split()
        if len(words) > 4:  # Only filter if sentence is long
            filtered_words = [word for word in words if word not in filler_words]
            if filtered_words:  # Don't return empty
                normalized = " ".join(filtered_words)
        
        return {
            "normalized": normalized.strip(),
            "original": original,
            "had_dialect": had_dialect,
            "changes_made": normalized != original
        }
    
    def normalize_simple(self, text: str) -> str:
        """Simple normalization without metadata"""
        result = self.normalize(text)
        return result["normalized"]

# Create singleton instance
normalizer = ArabicNormalizer()