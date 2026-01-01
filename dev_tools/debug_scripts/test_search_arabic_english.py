import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.chatbot_service import chatbot_service

uid = 'test_trans'
print('Search English: Argan Hair Oil')
r1 = chatbot_service.process_message(uid, 'Argan Hair Oil')
print(' -> intent:', r1.get('intent'))
print(' -> response:', r1.get('response'))
print('\nSearch Arabic title: أرغان شعر زيت (from token map)')
r2 = chatbot_service.process_message(uid, 'أرغان')
print(' -> intent:', r2.get('intent'))
print(' -> response:', r2.get('response'))
print('\nSearch Arabic description snippet: يرطب الشعر')
r3 = chatbot_service.process_message(uid, 'يرطب الشعر')
print(' -> intent:', r3.get('intent'))
print(' -> response:', r3.get('response'))
