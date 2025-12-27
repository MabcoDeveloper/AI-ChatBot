from services.chatbot_service import chatbot_service

print('Running price tests...')
res = chatbot_service.process_message(user_id='tester', message='شو هو ارخص شامبو')
print('Response:', res['response'])
if res.get('data') and res['data'].get('summaries'):
    for s in res['data']['summaries']:
        print(' -', s['name'], s['price'], s['currency'], 'in_stock' if s['in_stock'] else 'out')

res2 = chatbot_service.process_message(user_id='tester', message='اريد كريم سعره بين ال 10 وال 50')
print('\nResponse range:', res2['response'])
if res2.get('data') and res2['data'].get('summaries'):
    for s in res2['data']['summaries']:
        print(' -', s['name'], s['price'], s['currency'], 'in_stock' if s['in_stock'] else 'out')
else:
    print('No summaries for range test')
