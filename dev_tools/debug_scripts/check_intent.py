import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import IntentDetector
id = IntentDetector()
phrases = ['هل عندكم عروض؟', 'هل عندكم عروض', 'عروض', 'هل عندكم؟', 'هل عندكم شامبو؟']
for p in phrases:
    print(p, '->', id.detect(p))

# Extra checks for _contains_keyword
print('\n_contains_keyword checks:')
print("_contains_keyword('هل عندكم عروض؟', 'عروض') ->", id._contains_keyword('هل عندكم عروض؟', 'عروض'))
print("_contains_keyword('هل عندكم عروض', 'عروض') ->", id._contains_keyword('هل عندكم عروض', 'عروض'))
print("_contains_keyword('عروض', 'عروض') ->", id._contains_keyword('عروض', 'عروض'))
print("_contains_keyword('هل عندكم؟', 'عندكم') ->", id._contains_keyword('هل عندكم؟', 'عندكم'))

