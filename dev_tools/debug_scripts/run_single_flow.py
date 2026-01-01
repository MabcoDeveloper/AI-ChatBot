import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service

uid = 'testbuyer'
msgs = ['أريد شراء أحمر شفاه ماك', 'أحمد, 0501234567', 'نعم']
for msg in msgs:
    print('\n----')
    print('Sending:', msg)
    try:
        res = chatbot_service.process_message(uid, msg)
    except Exception as e:
        print('Exception:', e)
        break
    print('Result type:', type(res))
    print('Result repr:', repr(res))
    if isinstance(res, dict):
        print('Intent:', res.get('intent'), 'Response:', res.get('response'))
