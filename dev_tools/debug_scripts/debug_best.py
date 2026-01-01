from services.chatbot_service import chatbot_service

queries = ['ما هو أفضل منتج', 'ما هو أفضل شامبو', 'أفضل منتجات']
for q in queries:
    print('===', q)
    print('intent parse ->', chatbot_service.intent_detector.detect(q))
    parsed = chatbot_service._parse_price_range(q)
    print('price parsed ->', parsed)
    filters = chatbot_service._extract_filters_from_query(q)
    print('filters ->', filters)
    res = chatbot_service.process_message(user_id='debugbest', message=q)
    print('response ->', res['response'])
    print('data ->', res.get('data'))
    print('\n')
