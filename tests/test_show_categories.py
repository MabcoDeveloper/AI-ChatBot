from services.chatbot_service import chatbot_service


def test_show_categories_for_generic_query():
    uid = 'test_show_cat'
    res = chatbot_service.process_message(uid, 'ماهي المنتجات المتوفرة')
    assert res.get('data') and res['data'].get('categories'), "Expected categories in response data"
    assert ('فئات' in res['response']) or ('الفئات' in res['response']), "Expected response text to mention categories"
