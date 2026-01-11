import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
user='multitest'
queries = [
    'اريد منتج للعناية بالبشرة',
    'اريد علاج للشعر التالف',
    'اريد حل للشعر المتقصف',
    'اعاني من مشكلة تقصف'
]
for q in queries:
    res = svc.process_message(user, q)
    print('Query:', q)
    print('Response:')
    print(res.get('response'))
    products = res.get('data', {}).get('products') or []
    print('Products:', len(products))
    if products:
        for i,p in enumerate(res['data']['summaries'],1):
            print(i, p['name'], '-', p['price'], '-', p.get('match_field'))
    print('-'*40)
