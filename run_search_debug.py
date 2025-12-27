from services.mongo_service import mongo_service
print('filter search_products with category and price range:')
res = mongo_service.search_products(category='العناية بالبشرة', min_price=10, max_price=50)
print('results:', [(p.get('name'), p.get('price')) for p in res])
print('Count:', len(res))

print('\nsearch with query=None and category present:')
res2 = mongo_service.search_products(query=None, category='العناية بالبشرة', min_price=10, max_price=50)
print('results2:', [(p.get('name'), p.get('price')) for p in res2])
print('Count:', len(res2))

print('\nsearch without price filter:')
res3 = mongo_service.search_products(category='العناية بالبشرة')
print('results3:', [(p.get('name'), p.get('price')) for p in res3])
print('Count:', len(res3))
