import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.chatbot_service import ChatbotService
import pprint

s = ChatbotService()
user = 'test'
s.user_state[user] = {'awaiting_size_selection': {'product': {'product_id': 'p1', 'price_map': {'50ml': 15, '100ml': 25}, 'currency': 'USD'}, 'sizes': ['50ml', '100ml']}}
res = s.process_message(user, '2')
pp = pprint.pformat(res, indent=2, width=120, sort_dicts=False)
print(pp)
