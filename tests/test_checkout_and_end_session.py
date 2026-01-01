from services.chatbot_service import chatbot_service
from services.mongo_service import mongo_service


def test_checkout_confirms_and_ends_session():
    user_id = 'test_user_checkout'

    # Start a search to get a product
    res = chatbot_service.process_message(user_id=user_id, message='هل عندكم شامبو')
    assert res['intent'] in ('search', 'fallback') or res.get('data')
    assert chatbot_service.search_context.get(user_id)

    # Ask for details / choose
    res2 = chatbot_service.process_message(user_id=user_id, message='نعم')
    # Bot either asks to choose or shows details directly
    if 'أي واحد' in res2['response']:
        res3 = chatbot_service.process_message(user_id=user_id, message='1')
    else:
        res3 = res2

    assert 'هل تريد شراء هذا المنتج' in res3['response']
    assert res3['data'] and res3['data'].get('product')

    # User agrees to buy -> ask for name/phone
    res4 = chatbot_service.process_message(user_id=user_id, message='نعم')
    assert 'اسمك' in res4['response'] or 'رقم' in res4['response']

    # Provide customer info
    res5 = chatbot_service.process_message(user_id=user_id, message='ليلى, 0551231234')
    assert 'تم إضافة' in res5['response']
    assert res5['data'] and res5['data'].get('cart')
    cart = res5['data']['cart']

    # Now confirm checkout
    res6 = chatbot_service.process_message(user_id=user_id, message='نعم')
    assert 'تم تأكيد الطلب' in res6['response']
    assert res6['data'] and res6['data'].get('set_cookie')
    assert res6['data']['set_cookie']['max_age'] == 0

    # Verify cart in DB is marked checked_out
    db_cart = mongo_service.get_cart_by_id(cart['cart_id'])
    assert db_cart and db_cart.get('checked_out')

    # Cleanup
    mongo_service.delete_cart(cart['cart_id'])


def test_end_session_deletes_cart():
    user_id = 'test_user_end'

    # Start and add to cart
    res = chatbot_service.process_message(user_id=user_id, message='هل عندكم شامبو')
    assert chatbot_service.search_context.get(user_id)

    res2 = chatbot_service.process_message(user_id=user_id, message='نعم')
    if 'أي واحد' in res2['response']:
        res3 = chatbot_service.process_message(user_id=user_id, message='1')
    else:
        res3 = res2

    # Start buy
    res4 = chatbot_service.process_message(user_id=user_id, message='نعم')
    res5 = chatbot_service.process_message(user_id=user_id, message='باسل, 0554443332')
    assert res5['data'] and res5['data'].get('cart')
    cart = res5['data']['cart']

    # User ends session explicitly
    res6 = chatbot_service.process_message(user_id=user_id, message='خلاص')
    assert 'تم إنهاء الجلسة' in res6['response']
    assert res6['data'] and res6['data'].get('set_cookie')
    assert res6['data']['set_cookie']['max_age'] == 0

    # Cart should be deleted from DB
    db_cart = mongo_service.get_cart_by_id(cart['cart_id'])
    assert db_cart is None

    # memory should be cleared
    assert user_id not in chatbot_service.memory
