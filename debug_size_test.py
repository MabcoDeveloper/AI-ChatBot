from services.chatbot_service import ChatbotService

svc = ChatbotService()
user_id = 'test_user'

# Prepare two sample products
p1 = {'product_id':'p1', 'title':'زيت الأرغان للشعر', 'title_ar':'زيت الأرغان للشعر', 'name':'زيت الأرغان', 'price_map':{'50ml':12.0,'100ml':25.0}, 'currency':'USD', 'in_stock':True}
p2 = {'product_id':'p2', 'title':'زيت جوز الهند', 'title_ar':'زيت جوز الهند', 'name':'زيت جوز الهند', 'price_map':{'50ml':10.0,'100ml':18.0}, 'currency':'USD', 'in_stock':True}

# Simulate search context with summaries and products
svc.search_context[user_id] = {'summaries':[{'product_id':'p1','name':'زيت الأرغان'},{'product_id':'p2','name':'زيت جوز الهند'}], 'products':[p1,p2]}

# 1) user selects product 1
print('=== User: 1 ===')
res = svc.process_message(user_id, '1')
print(res['response'])
print('State:', svc.user_state.get(user_id))

# 2) user says: 'اشتري 1' -> wants to buy product 1 (should ask for sizes)
print('\n=== User: اشتري 1 ===')
res = svc.process_message(user_id, 'اشتري 1')
print(res['response'])
print('State:', svc.user_state.get(user_id))

# 3a) user selects size by typing '2' -> should pick 100ml for product1
print('\n=== User: 2 ===')
res = svc.process_message(user_id, '2')
print(res['response'])
print('State:', svc.user_state.get(user_id))

# Reset and repeat variant where after step 2 user types 'اشتري 2' (buy 2) which should be treated as size selection
svc.user_state[user_id] = {}  # clear
# re-run initial detail and awaiting size
svc.process_message(user_id, '1')
print('\n=== User: اشتري 1 (start buy again) ===')
svc.process_message(user_id, 'اشتري 1')
print('State after asking sizes:', svc.user_state.get(user_id))

print('\n=== User: اشتري 2 (size chosen with buy keyword) ===')
res = svc.process_message(user_id, 'اشتري 2')
print(res['response'])
print('State:', svc.user_state.get(user_id))
