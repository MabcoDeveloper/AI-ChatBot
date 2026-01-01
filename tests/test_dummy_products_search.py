from services.chatbot_service import chatbot_service
from services.mongo_service import mongo_service


def test_dummy_products_seed_and_detail():
    # Ensure the DummyProducts from data/assets were seeded
    p = mongo_service.get_product_by_name('Argan Hair Oil')
    assert p and p.get('price_map'), 'Expected Dummy product Argan Hair Oil with price_map'

    uid = 'dummy_test'
    # Ask the bot about the product (English title should be searchable)
    r = chatbot_service.process_message(uid, 'Argan Hair Oil')
    assert r['intent'] in ('search', 'detail', 'clarify'), f"Unexpected intent: {r['intent']}"

    # If the bot returned search summaries, choose the first one
    if r.get('data') and r['data'].get('summaries'):
        r2 = chatbot_service.process_message(uid, '1')
    else:
        r2 = r

    assert r2.get('data') and r2['data'].get('product'), 'Expected product details in response data'
    # Detail response should include size-based prices or at least the word for sizes/prices
    assert ('أحجام/أسعار' in r2['response']) or ('السعر' in r2['response']), 'Expected size/price information in the detail response'
