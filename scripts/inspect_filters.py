import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService
svc = ChatbotService()
q = 'اريد زيت للشعر'
print('Query:', q)
print('Extracted filters:', svc._extract_filters_from_query(q))
print('Parse price:', svc._parse_price_range(q))
