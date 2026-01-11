import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
user='quicktest1'
res = svc.process_message(user, 'اعاني من مشكلة تقصف')
print('Response:', res.get('response'))
if res.get('data') and res['data'].get('products'):
    for i,p in enumerate(res['data']['products'],1):
        print(i, p.get('name'), '-', p.get('min_price'), '-', p.get('in_stock'))
else:
    print('No products returned in data.products')
