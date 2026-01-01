import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service

flows = {
    'flow1': ['ابحث عن شامبو', '1', 'نعم', 'يوسف, 050222333', 'نعم'],
    'flow2': ['هل عندكم منتجات من ماك؟', 'تفاصيل 1', 'نعم', 'سارة, 0551234567', 'نعم'],
    'flow3': ['ابحث عن كريم مرطب', 'نيڤيا', 'تفاصيل', 'نعم', 'منى, 0509876543', 'نعم']
}

for uid, msgs in flows.items():
    print('\n' + '='*50)
    print(f"Starting flow for {uid}")
    for msg in msgs:
        res = chatbot_service.process_message(uid, msg)
        print('-'*40)
        print(f"User -> {msg}")
        if isinstance(res, tuple):
            print('Tuple response:', res)
            continue
        print(f"Intent: {res.get('intent')} (conf={res.get('intent_confidence')})")
        print(f"Bot: {res.get('response')}")
        if res.get('data'):
            print('Data keys:', list(res.get('data').keys()))
    print(f"End of flow {uid}")
