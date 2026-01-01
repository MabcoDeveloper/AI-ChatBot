import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.mongo_service import mongo_service

print('Checking translation fields in DB...')
q_title = {"$or": [{"title_ar": {"$exists": False}}, {"title_ar": ""}]}
q_desc = {"$or": [{"description_ar": {"$exists": False}}, {"description_ar": ""}]}
count_title = mongo_service.products.count_documents(q_title)
count_desc = mongo_service.products.count_documents(q_desc)
print('Products missing title_ar:', count_title)
print('Products missing description_ar:', count_desc)
# show up to 5 examples
for doc in mongo_service.products.find(q_title).limit(5):
    print(' -', doc.get('product_id') or doc.get('_id'), 'title:', doc.get('name') or doc.get('title'))
for doc in mongo_service.products.find(q_desc).limit(5):
    print(' -', doc.get('product_id') or doc.get('_id'), 'desc snippet:', (doc.get('description') or '')[:80])
