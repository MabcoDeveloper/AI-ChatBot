from services.mongo_service import mongo_service

p = mongo_service.products.find_one({'product_id':'DUMMY7'}, {'_id':0,'product_id':1,'title_ar':1,'description_ar':1})
print('DUMMY7:', p)

missing = []
for prod in mongo_service.products.find({}, {'_id':0,'product_id':1,'title_ar':1,'description_ar':1}):
    if not prod.get('title_ar') or not prod.get('description_ar'):
        missing.append(prod.get('product_id'))
print('Products missing Arabic fields count:', len(missing))
if missing:
    print('Missing sample:', missing[:10])
else:
    print('All products have Arabic fields')
