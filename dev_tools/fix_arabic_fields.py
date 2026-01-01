import re, os
from services.mongo_service import mongo_service

assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'assets', 'data.js')
text = open(assets_path, 'r', encoding='utf-8').read()

# Find all product _id occurrences
ids = [m.group(1) for m in re.finditer(r"\b_id\s*:\s*\"(\d+)\"", text)]
print('Found asset ids:', len(ids))
updated = 0
for sid in ids:
    idx = text.find(f'_id: "{sid}"')
    if idx == -1:
        continue
    # take a generous window after the id to find title_ar/description_ar
    window = text[idx: idx + 1000]

    title_ar = ''
    desc_ar = ''
    m_title = re.search(r'title_ar\s*:\s*"([^"]+)"', window)
    if m_title:
        title_ar = m_title.group(1)
    m_desc = re.search(r'description_ar\s*:\s*"([^"]+)"', window)
    if m_desc:
        desc_ar = m_desc.group(1)

    pid = f'DUMMY{sid}'
    if title_ar or desc_ar:
        upd = {}
        if title_ar:
            upd['title_ar'] = title_ar
        if desc_ar:
            upd['description_ar'] = desc_ar
        if upd:
            res = mongo_service.products.update_one({'product_id': pid}, {'$set': upd})
            if getattr(res, 'modified_count', 0) > 0:
                updated += 1
                print('Updated', pid, 'title_ar present:', bool(title_ar), 'desc_ar present:', bool(desc_ar))
            else:
                p = mongo_service.products.find_one({'product_id': pid}, {'_id':0})
                if p:
                    same_title = (p.get('title_ar') == title_ar) if title_ar else True
                    same_desc = (p.get('description_ar') == desc_ar) if desc_ar else True
                    if not (same_title and same_desc):
                        print('Attempted update but modified_count == 0 for', pid, '— check permissions or data')
                else:
                    print('Product', pid, 'not found in DB; skipping')

print('Total products updated with Arabic fields:', updated)