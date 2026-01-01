import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.mongo_service import mongo_service

query = 'Argan Hair Oil'
print('Searching for ->', query)
results = mongo_service.search_products(query)
print('Found:', len(results))
for i, p in enumerate(results[:10], 1):
    print(i, p.get('name') or p.get('title'), '-', p.get('min_price') or p.get('price'))
