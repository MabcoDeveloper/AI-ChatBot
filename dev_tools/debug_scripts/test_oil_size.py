import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service

uid='oiltest'
print('Search: زيوت')
r = chatbot_service.process_message(uid, 'ابحث عن زيوت')
print('search intent:', r.get('intent'))
if r.get('data'):
    print('summaries count:', len(r['data'].get('summaries', [])))

res = chatbot_service.process_message(uid, 'زيوت')
print('intent:', res.get('intent'))
print(res.get('response'))

print('\nChoose first product (1)')
r2 = chatbot_service.process_message(uid, '1')
print('detail intent:', r2.get('intent'))
print('detail response:\n', r2.get('response'))

print('\nSay yes (نعم) to buy')
r3 = chatbot_service.process_message(uid, 'نعم')
print('after yes intent:', r3.get('intent'))
print('after yes response:\n', r3.get('response'))

# If asked to choose size, pick '1'
if r3.get('intent') == 'clarify' and 'اختر' in (r3.get('response') or ''):
    print('\nSelecting size 1')
    r4 = chatbot_service.process_message(uid, '1')
    print('after size selection intent:', r4.get('intent'))
    print(r4.get('response'))

print('\nDone')
