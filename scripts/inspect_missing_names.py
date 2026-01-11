import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.mongo_service import mongo_service

missing = []
for p in mongo_service.products.find({}, {'_id':0,'product_id':1,'name':1,'title':1,'title_ar':1,'price':1,'price_map':1,'in_stock':1}).limit(500):
    name = p.get('title_ar') or p.get('name') or p.get('title')
    if not name:
        missing.append(p)

print(f'Found {len(missing)} docs missing a name/title/name_ar')
for p in missing[:50]:
    print(p)

print('\nTesting search_products(category="Hair Care", min_price=10, max_price=50)')
prods = mongo_service.search_products(query=None, category='Hair Care', min_price=10, max_price=50, limit=20)
print('Returned', len(prods), 'products')
for p in prods:
    print('--- product ---')
    print('product_id:', p.get('product_id'))
    print('name:', p.get('name'))
    print('title:', p.get('title'))
    print('title_ar:', p.get('title_ar'))
    print('min_price:', p.get('min_price'))
    print('price:', p.get('price'))
    print('price_map:', p.get('price_map'))
    print('in_stock:', p.get('in_stock'))
