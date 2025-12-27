from services.chatbot_service import chatbot_service
from services.mongo_service import mongo_service


def test_purchase_via_choice_and_confirm():
    user_id = 'test_user_choice'

    # Search for a category with multiple results
    res = chatbot_service.process_message(user_id=user_id, message='اريد كريم من نيفيا')
    assert res['intent'] == 'search' or res['data']
    assert chatbot_service.search_context.get(user_id)

    # User confirms they want details (short 'نعم')
    res2 = chatbot_service.process_message(user_id=user_id, message='نعم')
    # Bot may either ask to choose (if multiple results) or show details directly (if single)
    assert ('أي واحد' in res2['response']) or ('تفاصيل المنتج' in res2['response'])

    # If bot asked to choose, send numeric choice; otherwise we already have details
    if 'أي واحد' in res2['response']:
        res3 = chatbot_service.process_message(user_id=user_id, message='1')
    else:
        res3 = res2

    # Bot should show details and ask if they'd like to buy
    assert 'هل تريد شراء هذا المنتج' in res3['response']
    assert res3['data'] and res3['data'].get('product')

    # User says yes to buy
    res4 = chatbot_service.process_message(user_id=user_id, message='نعم')
    assert 'تزويدي باسمك' in res4['response'] or 'رقم هاتفك' in res4['response']

    # Provide name and phone
    res5 = chatbot_service.process_message(user_id=user_id, message='سعيد, 0509998887')
    assert 'تم إضافة' in res5['response']
    assert res5['data'] and res5['data'].get('cart')

    # cleanup
    mongo_service.carts.delete_one({'cart_id': res5['data']['cart']['cart_id']})
