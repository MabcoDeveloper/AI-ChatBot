from services.mongo_service import mongo_service

res = mongo_service.search_products(query=None, category=None, brand=None, min_price=10, max_price=100)
print('results:', [(p.get('name'), p.get('price')) for p in res])
print('count:', len(res))
