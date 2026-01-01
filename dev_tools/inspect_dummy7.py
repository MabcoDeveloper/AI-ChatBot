from services.mongo_service import mongo_service
p = mongo_service.products.find_one({'product_id':'DUMMY7'},{'_id':0})
print('DUMMY7 found:', bool(p))
if p:
    print('title_ar:', repr(p.get('title_ar')))
    print('description_ar:', repr(p.get('description_ar')))
    print('description_ar contains متقصف:', 'متقصف' in (p.get('description_ar') or ''))
    print('description_ar contains متقصفة:', 'متقصفة' in (p.get('description_ar') or ''))
    print('description_ar lower:', (p.get('description_ar') or '').lower())
