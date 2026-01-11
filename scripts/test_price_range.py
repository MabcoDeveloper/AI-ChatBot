import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
from services.mongo_service import mongo_service

svc = ChatbotService()
q = 'اريد زيت للشعر بين 10 و50'
print('Parsing price range...')
print(svc._parse_price_range(q))
print('Extracting filters...')
print(svc._extract_filters_from_query(q))
print('Running price handler...')
resp, data = svc._handle_price_intent(q)
print('Response:')
print(resp)
print('Data summaries count:', len(data['summaries']) if data and data.get('summaries') else 0)
print('Calling mongo search directly...')
parsed = svc._parse_price_range(q)
filters = svc._extract_filters_from_query(q)
prods = mongo_service.search_products(query=None, category=filters.get('category'), brand=filters.get('brand'), min_price=parsed.get('min'), max_price=parsed.get('max'))
print('Direct search products:', len(prods))
for p in prods[:10]:
    print('-', p.get('product_id') or p.get('name'), p.get('min_price'), p.get('max_price'), p.get('price'), p.get('price_map'))
