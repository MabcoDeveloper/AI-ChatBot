import re
import pytest

from services.chatbot_service import chatbot_service
from services.mongo_service import mongo_service


def test_buy_flow_creates_cart_and_sets_cookie():
    user_id = 'test_user_cart'

    # Search for a product to populate search_context
    search_msg = 'هل عندكم كريم نيڤيا'
    res = chatbot_service.process_message(user_id=user_id, message=search_msg)
    assert res['intent'] in ('search', 'fallback') or res['data']
    # Ensure search_context was set
    assert chatbot_service.search_context.get(user_id)

    # User indicates they want to buy (bot should ask for name/phone)
    buy_msg = 'أريد شراء هذا المنتج'
    res2 = chatbot_service.process_message(user_id=user_id, message=buy_msg)
    assert 'اسمك' in res2['response'] or 'رقم' in res2['response']

    # Provide name and phone (use simple digits)
    cust_msg = 'أحمد, 0501234567'
    res3 = chatbot_service.process_message(user_id=user_id, message=cust_msg)

    # Bot should confirm and return cart data and set_cookie directive
    assert 'تم إضافة' in res3['response']
    assert res3['data'] and res3['data'].get('cart')
    assert res3['data'].get('set_cookie')

    cart = res3['data']['cart']
    assert cart.get('items') and len(cart['items']) >= 1

    # Verify cart in DB
    db_cart = mongo_service.get_cart_by_customer('أحمد', '0501234567')
    assert db_cart is not None
    assert any(item['product_id'] for item in db_cart.get('items', []))

    # Cleanup: remove the created cart to avoid test pollution
    mongo_service.carts.delete_one({'cart_id': cart['cart_id']})
