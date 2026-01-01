import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service, IntentDetector

print('Running basic checks...')

idet = IntentDetector()
assert idet.detect('هل عندكم عروض؟')['intent'] == 'offers', 'Offers detection failed'
print('Offers detection: OK')

r = chatbot_service.process_message('rt1', 'مرحبا')
assert r['intent'] == 'greeting', 'Greeting failed'
print('Greeting flow: OK')

r = chatbot_service.process_message('rt2', 'بكم أحمر شفاه ماك؟')
assert r['intent'] in ('price', 'search'), 'Price/search detection failed'
print('Price query: OK')

# buy flow
user_id = 'rtflow'
r1 = chatbot_service.process_message(user_id, 'أريد شراء أحمر شفاه ماك')
assert r1['intent'] == 'buy'
r2 = chatbot_service.process_message(user_id, 'مريم, 0559876543')
assert 'تم إضافة' in r2['response']
r3 = chatbot_service.process_message(user_id, 'نعم')
assert 'تم تأكيد الطلب' in r3['response']
print('Buy flow: OK')

print('All basic checks passed')
