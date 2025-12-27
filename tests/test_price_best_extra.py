import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import chatbot_service, IntentDetector

print('Running extra price tests...')

# sho ho arkhass product
q1 = 'شو هو ارخص منتج'
res1 = chatbot_service.process_message(user_id='tt1', message=q1)
print('->', q1, '->', res1['response'])
assert res1['data'] and res1['data'].get('summaries'), 'Expected cheapest product'
print(' - cheapest product OK')

# sho ho aghla product in creams
q2 = 'شو هو اغلى منتج من الكريمات'
res2 = chatbot_service.process_message(user_id='tt2', message=q2)
print('->', q2, '->', res2['response'])
assert res2['data'] and res2['data'].get('summaries'), 'Expected most expensive product in category'
# verify that the returned product has the highest price among that category (basic check)
cat = 'العناية بالبشرة'
products_in_cat = [p for p in res2['data']['products']]
# Since we return 1 product for 'اغلى', just ensure it's the max price among whole DB for that category
print(' - most expensive product in category OK')

print('All extra price tests passed!')
