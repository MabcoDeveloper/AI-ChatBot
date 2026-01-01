from data import seed_products

parsed = seed_products._load_dummy_products_from_assets()
print('Parsed count:', len(parsed))
for p in parsed:
    if p.get('product_id') == 'DUMMY7':
        print('DUMMY7 parsed:', p)
        break
# Print first 3 parsed entries to inspect title_ar
for i, p in enumerate(parsed[:3], 1):
    print(i, p.get('product_id'), 'title_ar:', repr(p.get('title_ar')))
