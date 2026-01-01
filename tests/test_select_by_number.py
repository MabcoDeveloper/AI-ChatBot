from services.chatbot_service import chatbot_service


def test_select_number_simple():
    uid = 'sel1'
    # Clear conversation state
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    r1 = chatbot_service.process_message(uid, 'ابحث عن شامبو')
    assert r1['intent'] in ('search', 'best', 'fallback')
    data = r1.get('data')
    assert data and data.get('summaries'), 'Expected search summaries for شامبو'

    # Choose the first product by sending '1'
    r2 = chatbot_service.process_message(uid, '1')
    assert r2['intent'] in ('detail', 'clarify')
    assert r2.get('data') and r2['data'].get('product')


def test_select_number_with_extra_words():
    uid = 'sel2'
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    r1 = chatbot_service.process_message(uid, 'ابحث عن كريم مرطب')
    assert r1.get('data') and r1['data'].get('summaries')

    # Variations: '1 تفاصيل', 'رقم 1', Arabic-Indic digit '١'
    for msg in ['1 تفاصيل', 'رقم 1', '١']:
        r2 = chatbot_service.process_message(uid, msg)
        assert r2['intent'] in ('detail', 'clarify')
        assert r2.get('data') and r2['data'].get('product')
        # ensure last_viewed_product was stored in state
        st = chatbot_service.user_state.get(uid, {})
        assert st.get('last_viewed_product') and st['last_viewed_product'].get('product_id')


def test_purchase_by_number_direct():
    uid = 'buyby1'
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    r1 = chatbot_service.process_message(uid, 'ابحث عن شامبو')
    assert r1.get('data') and r1['data'].get('summaries')

    # Try 'اشتري 1' -> should start buy flow (ask for customer info or size)
    r2 = chatbot_service.process_message(uid, 'اشتري 1')
    assert r2['intent'] in ('buy', 'clarify')
    assert 'لإتمام الشراء' in r2['response'] or 'المنتج يحتوي على أحجام متعددة' in r2['response']


def test_select_with_embedded_number_sentence():
    uid = 'sel4'
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    r1 = chatbot_service.process_message(uid, 'ابحث عن كريم مرطب')
    assert r1.get('data') and r1['data'].get('summaries')

    # 'أريد الرقم 2 الآن' should be interpreted as selecting product 2 and returning details
    r2 = chatbot_service.process_message(uid, 'أريد الرقم 2 الآن')
    assert r2['intent'] in ('detail', 'clarify')
    assert r2.get('data') and r2['data'].get('product')


def test_recent_questions_memory():
    uid = 'recent1'
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    chatbot_service.process_message(uid, 'ابحث عن شامبو')
    chatbot_service.process_message(uid, 'ابحث عن كريم مرطب')
    chatbot_service.process_message(uid, 'ابحث عن زيوت')

    rs = chatbot_service.user_state.get(uid, {}).get('recent_questions', [])
    assert isinstance(rs, list)
    assert len(rs) <= 3
    # Last recent question should be the last query we sent
    assert rs[-1] == 'ابحث عن زيوت'


def test_select_out_of_range():
    uid = 'sel3'
    try:
        chatbot_service.clear_conversation(uid)
    except Exception:
        pass

    r1 = chatbot_service.process_message(uid, 'ابحث عن شامبو')
    assert r1.get('data') and r1['data'].get('summaries')

    # Choose an out-of-range number
    r2 = chatbot_service.process_message(uid, '99')
    assert r2['intent'] == 'clarify'
    assert 'لم أجد خيارًا' in r2['response']
