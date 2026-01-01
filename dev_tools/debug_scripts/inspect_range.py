from services.chatbot_service import chatbot_service
q = 'اريد كريم سعره بين ال 10 وال 50'
print('parse:', chatbot_service._parse_price_range(q))
print('filters:', chatbot_service._extract_filters_from_query(q))
