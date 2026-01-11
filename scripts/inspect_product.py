import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.mongo_service import mongo_service
p = mongo_service.products.find_one({'title_ar': 'زيت الأركان للشعر'})
print('Found:', bool(p))
if p:
    for k in ['product_id','title_ar','name','price','min_price','max_price','price_map','currency']:
        print(k,':', p.get(k))
    print('Description_ar:', p.get('description_ar'))
else:
    print('Not found')
