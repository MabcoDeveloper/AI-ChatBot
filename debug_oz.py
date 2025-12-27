from services.chatbot_service import chatbot_service

r = chatbot_service.process_message(user_id='tester_oz', message='ماهي الزيوت المتوفرة')
print('R1 response:', r.get('response'))
print('R1 data:', r.get('data'))

r2 = chatbot_service.process_message(user_id='tester_oz', message='نعم')
print('R2 response:', r2.get('response'))
print('R2 data:', r2.get('data'))

# Now multi-case
r3 = chatbot_service.process_message(user_id='tester_oz', message='عندكم منتجات في العناية بالبشرة؟')
print('R3 response:', r3.get('response'))

r4 = chatbot_service.process_message(user_id='tester_oz', message='نعم')
print('R4 response:', r4.get('response'))
print('R4 data:', r4.get('data'))
