import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
from services.mongo_service import mongo_service
from models.memory_manager import memory_manager

svc = ChatbotService()
user = 'test_user_buy2'
# Insert a fresh test product so we don't mix with previous test state
p={'product_id':'CUST_TEST3','name':'زيت تجريبي جديد','title':'زيت تجريبي جديد','title_ar':'زيت تجريبي جديد','description':'اختبار','price':None,'min_price':None,'price_map':{'50ml':12,'100ml':20},'currency':'SAR','in_stock':True}
mongo_service.products.insert_one(p)
# Set last_viewed_product in user state cleanly
st = {}
st['last_viewed_product'] = p
memory_manager.set_user_state(user, st)
# User says 'اشتري 1' (intending to buy the viewed product)
res = svc.process_message(user, 'اشتري 1')
print('Response:', res.get('response'))
print('State after:', memory_manager.get_user_state(user))
# Now supply name+phone (without choosing size) - should still ask for size if sizes exist
res2 = svc.process_message(user, 'ياسين ,0941702512')
print('Response to name+phone:', res2.get('response'))
print('State after 2:', memory_manager.get_user_state(user))
# Now choose size by number (2 -> 100ml)
res3 = svc.process_message(user, '2')
print('Response to size:', res3.get('response'))
print('State after 3:', memory_manager.get_user_state(user))
# Now provide name+phone to complete adding to cart
res4 = svc.process_message(user, 'ياسين ,0941702512')
print('Response to name+phone after size:', res4.get('response'))
print('Full response body:', res4)
print('Data keys:', list(res4.get('data', {}).keys()))
print('Final state:', memory_manager.get_user_state(user))
