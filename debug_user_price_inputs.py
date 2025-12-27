from services.chatbot_service import chatbot_service

inputs = [
    'اريد منتجات بين 10 و 100',
    'اريد منتجات سعرها بين 10 و 100',
]

for q in inputs:
    print('=== Query:', q)
    parsed = chatbot_service._parse_price_range(q)
    filters = chatbot_service._extract_filters_from_query(q)
    print('parsed:', parsed)
    print('filters:', filters)
    res = chatbot_service.process_message(user_id='tester', message=q)
    print('response:', res['response'])
    print('data:', res.get('data'))
    print('\n')
