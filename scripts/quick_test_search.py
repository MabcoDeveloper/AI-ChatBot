import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.mongo_service import mongo_service

res = mongo_service.search_products(query='اعاني من مشكلة تقصف', limit=10)
print('Found:', len(res))
for i,p in enumerate(res[:10],1):
    print(i, p.get('product_id') or p.get('name'), '-', p.get('min_price'), '-', p.get('in_stock'))
