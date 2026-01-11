import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import ChatbotService

svc = ChatbotService()
q = 'اريد زيت للشعر'
print('Query:', q)
resp, data = svc._handle_search_intent(q)
print('\nResponse:')
print(resp)
print('\nData keys:', list(data.keys()) if data else None)
if data and data.get('summaries'):
    print('Summaries count:', len(data['summaries']))
    for idx, s in enumerate(data['summaries'][:10], start=1):
        print(idx, s.get('name'), s.get('price'), s.get('in_stock'))
else:
    print('No product summaries')
