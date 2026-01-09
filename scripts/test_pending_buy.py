import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.chatbot_service import ChatbotService
import pprint

s = ChatbotService()
user = 'test2'
# simulate pending_buy with sizes attached
s.user_state[user] = {'pending_buy': {'product': {'product_id': 'p1','name':'Test','price_map':{'50ml':15,'100ml':25},'currency':'USD'}, 'awaiting': 'size_selection', 'sizes':['50ml','100ml']}}
res = s.process_message(user, '1')
print('response1:')
pprint.pprint(res)

# simulate pending_buy with no sizes but product includes price_map
s.user_state[user] = {'pending_buy': {'product': {'product_id': 'p1','name':'Test','price_map':{'50ml':15,'100ml':25},'currency':'USD'}, 'awaiting': 'size_selection'}}
res2 = s.process_message(user, '2')
print('\nresponse2:')
pprint.pprint(res2)
