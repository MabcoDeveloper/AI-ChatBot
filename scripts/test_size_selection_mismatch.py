import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
from models.memory_manager import memory_manager
from services.mongo_service import mongo_service

svc = ChatbotService()
user = 'test_size_mismatch'
# Insert a test product
p = {'product_id': 'SIZE_TEST1', 'name': 'اختبار أحجام', 'title_ar': 'اختبار أحجام', 'description_ar': 'حجم 150ml لحالة A، حجم 300ml لحالة B', 'price_map': {'150ml': 24, '300ml': 42}, 'currency': 'SAR', 'in_stock': True}
mongo_service.products.insert_one(p)
# set user state to be awaiting size selection
st = {}
st['awaiting_size_selection'] = {'product': p, 'sizes': ['150ml','300ml']}
st['pending_buy'] = {'product': {'product_id': p['product_id'], 'name': p['name']}, 'awaiting': 'size_selection'}
memory_manager.set_user_state(user, st)
# set search_context summaries to a single-item list to simulate unrelated prior summary
svc.search_context[user] = [{'product_id': 'OTHER1', 'name': 'غير متعلق'}]
# Now send '2' and expect it to be interpreted as selecting the second size
res = svc.process_message(user, '2')
print('Response:', res.get('response'))
print('Data:', res.get('data'))
assert res.get('data', {}).get('size') == '300ml', 'Expected selected size to be 300ml'
print('✅ Size-selection mismatch test passed')
