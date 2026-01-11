import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
from services.mongo_service import mongo_service
from models.memory_manager import memory_manager

svc = ChatbotService()
user = 'test_user_integration_no_pid'
# Insert product WITHOUT explicit product_id (as in the provided data structure)
p = {
    'title': 'Argan Hair Oil',
    'title_ar': 'زيت الأركان للشعر',
    'price': {'50ml': 15, '100ml': 25, '200ml': 40},
    'description': 'Rich argan oil for nourishing and moisturizing dry hair.',
    'description_ar': 'زيت أركان غني لتغذية وترطيب الشعر الجاف.',
    'category': 'Hair Care',
    'type': 'oil',
    'size': ['50ml', '100ml', '200ml'],
    'in_stock': True
}
res_ins = mongo_service.products.insert_one(p)
# Sanity check: ensure the inserted doc has an ObjectId
assert res_ins.inserted_id is not None
# Clear user state and proceed through flow
memory_manager.set_user_state(user, {})
# Start by searching for "اريد زيت" which should return our product in summaries
r = svc.process_message(user, 'اريد زيت')
# select the summary that matches our title
summaries = r.get('data', {}).get('summaries') or []
found_idx = None
for i, s in enumerate(summaries):
    if 'الأركان' in (s.get('name') or s.get('title') or ''):
        found_idx = i
        break
assert found_idx is not None, 'Could not find inserted product in summaries'
sel = str(found_idx + 1)
r_detail = svc.process_message(user, sel)  # view detail
print('After detail view state:', memory_manager.get_user_state(user))
print('Detail response data keys:', list(r_detail.get('data', {}).keys()) if r_detail.get('data') else None)
print('Detail product object:', r_detail.get('data', {}).get('product'))
r_buy = svc.process_message(user, 'اشتري 1')  # start buy
print('After buy intent state:', memory_manager.get_user_state(user))
r_size = svc.process_message(user, '1')  # choose size 50ml
print('After size selection state:', memory_manager.get_user_state(user))
r_confirm = svc.process_message(user, 'نعم')  # confirm
print('Confirm data payload:', r_confirm.get('data'))
assert r_confirm.get('data', {}).get('product_title') is not None, 'product_title should be present for product with no explicit product_id'
# backward-compat alias: 'size' should be present and match selected size
assert r_confirm.get('data', {}).get('size') == '50ml', f"expected size '50ml', got {r_confirm.get('data', {}).get('size')!r}"
# product_id should be filled (never None) and be a string (normalized)
pid = r_confirm.get('data', {}).get('product_id')
assert pid is not None, 'product_id should be present and normalized even if the doc had no explicit product_id'
assert isinstance(pid, str), f'product_id should be a string after normalization, got {type(pid)}'
# responses should be JSON serializable (sanitizer applied)
import json
json.dumps(r_confirm)

# cleanup
mongo_service.products.delete_one({'_id': res_ins.inserted_id})
memory_manager.set_user_state(user, {})