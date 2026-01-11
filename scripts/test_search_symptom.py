import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
user = 'test_symptom'
res = svc.process_message(user, 'اعاني من مشكلة تقصف')
print('Response message:')
print(res.get('response'))
products = res.get('data', {}).get('products') or []
print('Products found:', len(products))
assert len(products) > 0, 'Expected at least one product for تقصف'
# Check description contains token if available
found = False
for p in products:
    desc = (p.get('description_ar') or '').lower()
    if 'تقصف' in desc or 'متقصف' in desc:
        found = True
        break
print('Symptom token found in description:', found)
assert found, 'No product description contains تقصف or متقصف'
# Ensure the response indicates description match when available
first_summary = res.get('data', {}).get('summaries', [])[0]
print('First summary match_field:', first_summary.get('match_field'))
assert first_summary.get('match_field') == 'description_ar', 'Expected first matched product to be marked as description_ar'
print('✅ Symptom search test passed')
