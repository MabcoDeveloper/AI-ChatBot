from services.chatbot_service import chatbot_service, IntentDetector


def test_offers_detection_with_punctuation():
    idet = IntentDetector()
    res = idet.detect('هل عندكم عروض؟')
    assert res['intent'] == 'offers'


def test_greeting_intent_and_response():
    res = chatbot_service.process_message('t1', 'مرحبا')
    assert res['intent'] == 'greeting'
    assert 'مساعد' in res['response']


def test_price_query_detects_price_and_returns_products():
    res = chatbot_service.process_message('t2', 'بكم أحمر شفاه ماك؟')
    assert res['intent'] in ('price', 'search')
    assert 'أحمر شفاه' in res['response'] or (res.get('data') and res['data'].get('products'))


def test_buy_flow_end_to_end():
    user_id = 'testflow'
    # start buy
    r1 = chatbot_service.process_message(user_id, 'أريد شراء أحمر شفاه ماك')
    assert r1['intent'] == 'buy'
    # provide name & phone
    r2 = chatbot_service.process_message(user_id, 'سارة, 0551234567')
    assert r2['intent'] == 'buy'
    assert 'تم إضافة' in r2['response'] or r2.get('data')
    # confirm purchase
    r3 = chatbot_service.process_message(user_id, 'نعم')
    assert r3['intent'] == 'buy'
    assert 'تم تأكيد الطلب' in r3['response']
