from services.mongo_service import MongoDBService
import re
m = MongoDBService()
q = 'شعري متقصف'

def _simplify_ar(s: str) -> str:
    if not s:
        return ''
    v = s.lower()
    mappings = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': '', 'ڤ': 'ف'}
    for a, b in mappings.items():
        v = v.replace(a, b)
    v = re.sub(r'[^0-9\u0600-\u06FF\s]', '', v)
    v = re.sub(r'\s+', ' ', v).strip()
    return v

print('query:', q)
print('query simplified:', _simplify_ar(q))

p = m.products.find_one({'product_id': '7'}, {'_id': 0, 'product_id': 1, 'title_ar': 1, 'description_ar': 1})
print('\nProduct 7:')
print(' raw title_ar:', p.get('title_ar'))
print('simpl title_ar:', _simplify_ar(p.get('title_ar')))
print(' raw description_ar:', p.get('description_ar'))
print('simpl description_ar:', _simplify_ar(p.get('description_ar')))

# Show token regex candidate pass
tokens = [t for t in re.split(r'\s+', q) if t.strip()]
tokens_s = [_simplify_ar(t) for t in tokens]
print('\ntokens:', tokens, '=> simplified:', tokens_s)
pattern = '|'.join([re.escape(t) for t in set(tokens_s)])
print('pattern used to find candidates:', pattern)
print('\nCandidates matching any token:')
for doc in m.products.find({'$or': [{'title_ar': {'$regex': pattern, '$options': 'i'}}, {'description_ar': {'$regex': pattern, '$options': 'i'}}]}, {'_id':0,'product_id':1,'title_ar':1,'description_ar':1}):
    print(doc)
