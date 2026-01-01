from services.mongo_service import mongo_service

q = 'شعري متقصف'
print('Query:', q)
# list any products whose description_ar contains 'متقصف' (case-insensitive) or title_ar contains 'متقصف'
matches = list(mongo_service.products.find({'$or':[{'description_ar':{'$regex':'متقصف','$options':'i'}},{'title_ar':{'$regex':'متقصف','$options':'i'}}]}, {'_id':0,'product_id':1,'title_ar':1,'description_ar':1,'category_ar':1}))
print('Products with متقصف in Arabic fields:', len(matches))
for p in matches:
    print(p.get('product_id'), p.get('title_ar'), p.get('category_ar'))

p7 = mongo_service.products.find_one({'product_id':'DUMMY7'},{'_id':0})
print('\nDUMMY7 exists:', bool(p7))
if p7:
    for k in ['product_id','title_ar','description_ar','category_ar','title']:
        print(k+':', p7.get(k))
