from services.chatbot_service import chatbot_service
from services.mongo_service import mongo_service


def test_arabic_type_search_oils():
    uid = 'catsearch1'
    r = chatbot_service.process_message(uid, 'ماهي الزيوت المتوفرة')
    assert r['intent'] in ('search', 'detail', 'clarify', 'fallback'), f"Unexpected intent: {r['intent']}"
    data = r.get('data')
    assert data and data.get('summaries'), 'Expected search summaries for "زيوت"'
    # At least one returned product should have an attributes.type containing 'oil' or category Hair Care
    found = False
    for s in data.get('summaries', []):
        pid = s.get('product_id')
        p = mongo_service.get_product_by_id(pid) if pid else None
        if p:
            t = (p.get('attributes') or {}).get('type') or p.get('type') or ''
            cat = p.get('category') or ''
            if 'oil' in str(t).lower() or 'hair care' in str(cat).lower():
                found = True
                break
    assert found, 'Expected at least one oil or Hair Care product in results'