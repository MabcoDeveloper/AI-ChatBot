import json
from data import seed_products
prods = seed_products._load_dummy_products_from_assets()
print('Parsed dummy products:', len(prods))
for p in prods:
    if p.get('product_id') == 'DUMMY3':
        print(json.dumps(p, ensure_ascii=False, indent=2))
        break
else:
    print('DUMMY3 not found')
