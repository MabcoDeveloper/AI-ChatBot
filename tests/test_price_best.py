import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import chatbot_service
from services.chatbot_service import IntentDetector

print('Running price & best tests...')

# Test parsing of numeric ranges
p = chatbot_service._parse_price_range('اريد منتجات بين 10 و 100')
assert p['min'] == 10.0 and p['max'] == 100.0 and not p['cheapest'], f"Range parse failed: {p}"
print(' - price range parsing OK')

# Test IntentDetector numeric detection -> price
det = IntentDetector()
res = det.detect('اريد منتجات بين 10 و 100')
assert res['intent'] == 'price', f"Intent not detected as price: {res}"
print(' - intent detector numeric range -> price OK')

# Test processing a general price range query returns products within range
resp = chatbot_service.process_message(user_id='test1', message='اريد منتجات بين 10 و 100')
assert resp['data'] and resp['data'].get('summaries'), 'Expected summaries for price range query'
for s in resp['data']['summaries']:
    price = s.get('price')
    assert price is None or (10.0 <= price <= 100.0), f"Product price out of requested range: {s}"
print(' - general price range search OK')

# Test cheapest/product specific queries
resp2 = chatbot_service.process_message(user_id='test2', message='شو هو ارخص شامبو')
assert resp2['data'] and resp2['data'].get('summaries'), 'Expected cheapest shampoo result'
print(' - cheapest shampoo OK')

# Test best product (general)
resp3 = chatbot_service.process_message(user_id='test3', message='ما هو أفضل منتج')
assert resp3['data'] and resp3['data'].get('summaries'), 'Expected best product result'
best_name = resp3['data']['summaries'][0]['name']
print(' - best product (general) OK:', best_name)

# Test best product for category
resp4 = chatbot_service.process_message(user_id='test4', message='ما هو أفضل شامبو')
assert resp4['data'] and resp4['data'].get('summaries'), 'Expected best shampoo result'
print(' - best product by category OK:', resp4['data']['summaries'][0]['name'])

print('\nAll price & best tests passed!')
