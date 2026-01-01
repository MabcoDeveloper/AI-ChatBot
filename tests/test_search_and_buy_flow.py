from services.chatbot_service import chatbot_service


def test_search_shampoo_and_view_details():
    uid = 'search1'
    r1 = chatbot_service.process_message(uid, 'ابحث عن شامبو')
    assert r1['intent'] in ('search', 'best', 'fallback')
    # Expect at least one product summary in data
    data = r1.get('data')
    assert data and data.get('summaries'), 'Expected search summaries for شامبو'

    # Choose the first product by sending '1'
    r2 = chatbot_service.process_message(uid, '1')
    assert r2['intent'] in ('detail', 'clarify')
    assert 'تفاصيل' in r2['response'] or 'تفاصيل المنتج' in r2['response'] or 'هل تريد شراء' in r2['response']


def test_filter_by_brand_and_buy_flow():
    uid = 'buyer1'
    # Ask for brand-specific products
    r1 = chatbot_service.process_message(uid, 'هل عندكم منتجات من ماك؟')
    assert r1['intent'] in ('search', 'best', 'offers')
    assert r1.get('data') and r1['data'].get('summaries'), 'Expected products for brand ماك'

    # Ask for details of the first item
    r2 = chatbot_service.process_message(uid, 'تفاصيل 1')
    assert r2['intent'] in ('detail', 'clarify')

    # Say yes to buy (may result in a size-clarify step if multiple sizes exist)
    r3 = chatbot_service.process_message(uid, 'نعم')
    if r3['intent'] == 'clarify':
        # choose the first size
        r3b = chatbot_service.process_message(uid, '1')
        assert r3b['intent'] == 'buy'
        r_after_confirm = r3b
    else:
        assert r3['intent'] == 'buy'
        r_after_confirm = r3

    # Provide customer info
    r4 = chatbot_service.process_message(uid, 'ليلى, 0550001111')
    assert r4['intent'] == 'buy' and ('تم إضافة' in r4['response'] or r4.get('data'))

    # Confirm purchase
    r5 = chatbot_service.process_message(uid, 'نعم')
    assert r5['intent'] == 'buy'
    assert 'تم تأكيد الطلب' in r5['response']


def test_follow_up_search_refinement():
    uid = 'search2'
    r1 = chatbot_service.process_message(uid, 'ابحث عن كريم مرطب')
    assert r1.get('data') and r1['data'].get('summaries')

    # User refines search by asking for brand 'نيڤيا'
    r2 = chatbot_service.process_message(uid, 'نيڤيا')
    assert r2['intent'] in ('search', 'best', 'price')
    assert r2.get('data') and (r2['data'].get('summaries') or r2['data'].get('clarify_options'))


def test_direct_customer_info_after_details():
    uid = 'directinfo'
    # Search and narrow to a single product
    r1 = chatbot_service.process_message(uid, 'ابحث عن كريم مرطب')
    r2 = chatbot_service.process_message(uid, 'نيڤيا')
    # Ask for details (will prompt which one or show product)
    r3 = chatbot_service.process_message(uid, 'تفاصيل')
    # Confirm detail selection if needed
    if r3['intent'] == 'clarify' or (isinstance(r3.get('response'), str) and r3.get('response', '').startswith('لم')):
        r3b = chatbot_service.process_message(uid, '1')
    else:
        r3b = r3
    assert r3b.get('data') and r3b['data'].get('product')

    # Now user provides name+phone directly without saying 'نعم'
    r4 = chatbot_service.process_message(uid, 'هند, 055222333')
    assert r4['intent'] == 'buy' and ('تم إضافة' in r4['response'] or r4.get('data'))

    # Confirm purchase
    r5 = chatbot_service.process_message(uid, 'نعم')
    assert r5['intent'] == 'buy' and 'تم تأكيد الطلب' in r5['response']


def test_size_selection_flow():
    uid = 'sizetest'
    r1 = chatbot_service.process_message(uid, 'ابحث عن زيوت')
    assert r1.get('data') and r1['data'].get('summaries')

    r2 = chatbot_service.process_message(uid, 'زيوت')
    # Choose the first product by sending '1'
    r3 = chatbot_service.process_message(uid, '1')
    assert r3['intent'] in ('detail', 'clarify')

    # Confirm buy intent
    r4 = chatbot_service.process_message(uid, 'نعم')
    # Should ask to choose size when multiple sizes are available
    assert r4['intent'] == 'clarify' and ('حجم' in (r4.get('response') or '') or 'أحجام' in (r4.get('response') or ''))

    # Pick the first size
    r5 = chatbot_service.process_message(uid, '1')
    assert r5['intent'] == 'buy' and 'يرجى تزويدي' in (r5.get('response') or '')

    # Provide customer info and confirm
    r6 = chatbot_service.process_message(uid, 'سارة, 050333444')
    assert r6['intent'] == 'buy' and ('تم إضافة' in r6['response'] or r6.get('data'))

    r7 = chatbot_service.process_message(uid, 'نعم')
    assert r7['intent'] == 'buy' and 'تم تأكيد' in r7['response']

