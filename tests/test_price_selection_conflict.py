import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
from services.mongo_service import mongo_service

svc = ChatbotService()
user = 'test_user_price_conflict'

# Ensure there is a prior search_context present to simulate previous interaction
svc.search_context[user] = [{'index': 1, 'product_id': 'dummy', 'name': 'X'}]

# Send a price-range message that includes numbers and should be treated as a price query, not a selection
q = 'اريد منتجات سعرهن بين ال 10 و 50'
r = svc.process_message(user, q)
print('Response intent:', r.get('intent'))
print('Response data keys:', list(r.get('data', {}).keys()) if r.get('data') else None)
assert r.get('intent') == 'price' or (r.get('data') and 'summaries' in r.get('data')) , 'Price-range query was incorrectly treated as a selection'

# cleanup
svc.search_context.pop(user, None)
