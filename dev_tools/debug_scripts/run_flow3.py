import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service

uid = 'flow3'
msgs = ['ابحث عن كريم مرطب', 'نيڤيا', 'تفاصيل', 'نعم', 'منى, 0509876543', 'نعم']
for msg in msgs:
    res = chatbot_service.process_message(uid, msg)
    print('---')
    print('User:', msg)
    print('Intent:', res.get('intent'), 'Confidence:', res.get('intent_confidence'))
    print('Response:', res.get('response'))
    if res.get('data'):
        print('Data keys:', list(res.get('data').keys()))
