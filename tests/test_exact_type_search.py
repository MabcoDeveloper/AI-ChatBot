import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
user = 'test_user_exact_type'

# Query for multi-word type 'زيت جوز الهند' which should return a specific product not a generic category list
q = 'زيت جوز الهند'
r = svc.process_message(user, q)
print('Intent:', r.get('intent'))
print('Response:', r.get('response'))
assert not (r.get('response') and r.get('response').startswith('وجدت') and 'فئة' in r.get('response')), 'Exact type query incorrectly treated as generic category browse'
# Expect summaries to include the exact product name
summaries = r.get('data', {}).get('summaries') or []
assert any('جوز' in (s.get('name') or '') for s in summaries), 'Expected at least one summary referencing "جوز" for coconut oil'

# cleanup
svc.search_context.pop(user, None)
