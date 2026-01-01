from services.mongo_service import mongo_service
q = 'شعري متقصف'
res = mongo_service.search_products(q, limit=5)
print('Top results for', q)
for i, r in enumerate(res, 1):
    print(i, r.get('product_id'), r.get('title_ar'))
