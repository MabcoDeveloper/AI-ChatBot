from services.chatbot_service import chatbot_service

user_id = 'dbg_user_buy'
print('1) Search')
res = chatbot_service.process_message(user_id=user_id, message='هل عندكم كريم نيڤيا')
print(res['intent'])
print(res['response'])
print('search data:', bool(res.get('data')))

print('\n2) Buy')
res2 = chatbot_service.process_message(user_id=user_id, message='أريد شراء هذا المنتج')
print(res2['intent'])
print(res2['response'])
print('data:', res2.get('data'))
print('\n3) Provide customer info')
res3 = chatbot_service.process_message(user_id=user_id, message='أحمد, 0501234567')
print(res3['intent'])
print(res3['response'])
print('data:', res3.get('data'))
