from services.mongo_service import mongo_service

print('Inspecting products in category العناية بالبشرة')
for p in mongo_service.products.find({'category': {'$regex': 'العناية بالبشرة', '$options': 'i'}}, {'_id':0,'name':1,'price':1}):
    print(p)

print('\nAll products:')
for p in mongo_service.products.find({}, {'_id':0,'name':1,'price':1,'category':1}):
    print(p)
