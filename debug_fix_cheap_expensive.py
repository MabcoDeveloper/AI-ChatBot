from services.chatbot_service import chatbot_service

queries = ['شو هو ارخص منتج', 'شو هو اغلى منتج من الكريمات']
for q in queries:
    print('===', q)
    print('intent:', chatbot_service.intent_detector.detect(q))
    print('_parse_price_range:', chatbot_service._parse_price_range(q))
    print('_extract_filters_from_query:', chatbot_service._extract_filters_from_query(q))
    r = chatbot_service.process_message(user_id='debug', message=q)
    print('response ->', r['response'])
    print('data ->', None if not r.get('data') else r['data'].get('summaries'))
    print()
