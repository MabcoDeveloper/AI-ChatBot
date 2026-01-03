from services.chatbot_service import ChatbotService

cb = ChatbotService()
q1 = "شو هي المنتتجات المتوفرة"
q2 = "ماهي المنتجات المتوفرة"

r1 = cb._handle_search_intent(q1)
r2 = cb._handle_search_intent(q2)

print('Q1 response:')
print(r1[0])
print('Q1 categories exists:', isinstance(r1[1], dict) and bool(r1[1].get('categories')))
print('------')
print('Q2 response:')
print(r2[0])
print('Q2 categories exists:', isinstance(r2[1], dict) and bool(r2[1].get('categories')))
