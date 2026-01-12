import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
user = 'test_user_smell_search'

# Simulate query that includes type word and a descriptive qualifier
q = 'بدي شامبو ريحتو حلوه'
r = svc.process_message(user, q)
print('Intent:', r.get('intent'))
print('Response:', r.get('response'))
# Ensure we did not treat the query as a generic category listing
assert not (r.get('response') and r.get('response').startswith('وجدت') and 'فئة' in r.get('response')), 'Query incorrectly treated as generic category browse'
# Ensure summaries exist and include at least one shampoo-related name when available
summaries = r.get('data', {}).get('summaries') or []
assert summaries, 'Expected search summaries for smell-based shampoo query'
assert any('شامبو' in (s.get('name') or '') for s in summaries), 'Expected at least one summary with "شامبو" in the name'

# cleanup
svc.search_context.pop(user, None)
