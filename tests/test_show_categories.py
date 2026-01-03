import os, sys
# Ensure the project root is on sys.path when this test is run directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services.chatbot_service import chatbot_service


def test_show_products_grouped_by_category_for_generic_query():
    uid = 'test_show_cat'
    res = chatbot_service.process_message(uid, 'ماهي المنتجات المتوفرة')
    assert res.get('data') and res['data'].get('categories'), "Expected categories in response data"

    cats = res['data']['categories']
    assert isinstance(cats, list) and len(cats) > 0, "Expected a non-empty list of category groups"

    # Each group should have category name, count and products list
    sample = cats[0]
    assert 'category' in sample and 'count' in sample and 'products' in sample, "Each category group should include 'category', 'count', and 'products'"
    assert isinstance(sample['products'], list), "Products for each category must be a list"

    # If there are products in DB, at least one category should include at least one product summary
    any_with_products = any(len(g.get('products', [])) > 0 for g in cats)
    assert any_with_products, "Expected at least one category to contain products in the grouped response"

    # Verify that the category label returned matches the product's Arabic category when available
    from services.mongo_service import mongo_service
    for g in cats:
        prods = g.get('products', [])
        if prods:
            pid = prods[0].get('product_id')
            # try to find product doc by _id or product_id
            doc = mongo_service.products.find_one({'$or': [{'_id': pid}, {'product_id': pid}]})
            if doc:
                expected_label = doc.get('category_ar') or doc.get('category')
                assert g['category'] == expected_label, f"Expected category label to be Arabic label from DB for group {g['category']}"

    # Response message should mention products or categories
    assert ('المنتجات' in res['response']) or ('الفئات' in res['response']) or ('منتجات' in res['response']), "Expected response text to mention products or categories"
