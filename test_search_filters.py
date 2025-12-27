# test_search_filters.py
"""Quick manual tests for category and brand search and chatbot behavior"""
from services.mongo_service import mongo_service
from services.chatbot_service import chatbot_service


def run_tests():
    print("Running search filter tests...")

    res1 = mongo_service.search_products(category="العناية بالبشرة")
    print(f"Search by exact category 'العناية بالبشرة' -> {len(res1)} results")

    res2 = mongo_service.search_products(category="العناية")
    print(f"Search by partial category 'العناية' -> {len(res2)} results")

    res3 = mongo_service.search_products(brand="ماك")
    print(f"Search by brand 'ماك' -> {len(res3)} results")

    res4 = mongo_service.search_products(brand="ما")
    print(f"Search by partial brand 'ما' -> {len(res4)} results")

    res5 = mongo_service.search_products(query="شامبو")
    print(f"Text search 'شامبو' -> {len(res5)} results")

    # Chatbot-level tests
    # Debug filters
    print('Filters for "ماهي منتجات نيفيا الموجودة":', chatbot_service._extract_filters_from_query('ماهي منتجات نيفيا الموجودة'))
    c1 = chatbot_service.process_message(user_id="tester", message="ماهي منتجات نيفيا الموجودة")
    print("Chatbot response for 'ماهي منتجات نيفيا الموجودة':")
    print(c1["response"])
    if c1.get("data") and c1["data"].get("summaries"):
        print("Summaries:")
        for s in c1["data"]["summaries"]:
            print(f" - {s['name']} | {s['price']} {s['currency']} | {'متوفر' if s['in_stock'] else 'غير متوفر'}")

    # Fuzzy product name search (typo)
    c_missp = chatbot_service.process_message(user_id="tester", message="ابحث عن سيروم فيتمين")
    print("Chatbot response for fuzzy 'ابحث عن سيروم فيتمين':")
    print(c_missp["response"])
    if c_missp.get("data") and c_missp["data"].get("summaries"):
        print("Fuzzy summaries:")
        for s in c_missp["data"]["summaries"]:
            print(f" - {s['name']} | {s['price']} {s['currency']} | {'متوفر' if s['in_stock'] else 'غير متوفر'}")

    print('Filters for "عندكم منتجات في العناية بالبشرة؟":', chatbot_service._extract_filters_from_query('عندكم منتجات في العناية بالبشرة?'))
    c2 = chatbot_service.process_message(user_id="tester", message="عندكم منتجات في العناية بالبشرة؟")
    print("Chatbot response for 'عندكم منتجات في العناية بالبشرة؟':")
    print(c2["response"])
    if c2.get("data") and c2["data"].get("products"):
        print(f"Products returned: {len(c2['data']['products'])}")

    # Follow-up: ask for details about the moisturizer (should use context)
    c3 = chatbot_service.process_message(user_id="tester", message="نعم تفاصيل الكريم المرطب")
    print("Chatbot response for follow-up 'نعم تفاصيل الكريم المرطب':")
    print(c3["response"])
    if c3.get("data") and c3["data"].get("product"):
        p = c3["data"]["product"]
        print(f"Detail product: {p.get('name')} | {p.get('price')} {p.get('currency')} | {'متوفر' if p.get('in_stock') else 'غير متوفر'}")

    # Closing / thanks behavior
    c_thanks = chatbot_service.process_message(user_id="tester", message="شكراً")
    print("Chatbot response for 'شكراً':")
    print(c_thanks["intent"], c_thanks["response"])

    c_thanks_en = chatbot_service.process_message(user_id="tester", message="thank you")
    print("Chatbot response for 'thank you':")
    print(c_thanks_en["intent"], c_thanks_en["response"])

    # Clarification behavior when no products found
    c4 = chatbot_service.process_message(user_id="tester", message="ماهي الزيوت المتوفرة")
    print("Chatbot response for 'ماهي الزيوت المتوفرة':")
    print(c4["response"])
    if c4.get("data") and c4["data"].get("clarify_options"):
        print("Clarify options:", c4["data"]["clarify_options"])

    # If the bot found exactly 1 product, a 'نعم' should return details for that product
    c_yes = chatbot_service.process_message(user_id="tester", message="نعم")
    print("Chatbot response for follow-up 'نعم' (single product case):")
    print(c_yes["response"])
    if c_yes.get("data") and c_yes["data"].get("product"):
        p = c_yes["data"]["product"]
        print(f"Detail product: {p.get('name')} | {p.get('price')} {p.get('currency')} | {'متوفر' if p.get('in_stock') else 'غير متوفر'}")

    # For multi-item results, 'نعم' should ask to choose which item
    # First trigger a multi-item context
    chatbot_service.process_message(user_id="tester", message="عندكم منتجات في العناية بالبشرة؟")
    c_yes_multi = chatbot_service.process_message(user_id="tester", message="نعم")
    print("Chatbot response for follow-up 'نعم' (multi product case):")
    print(c_yes_multi["response"])

    # Price queries: cheapest shampoo
    c_price = chatbot_service.process_message(user_id="tester", message="شو هو ارخص شامبو")
    print("Chatbot response for 'شو هو ارخص شامبو':")
    print(c_price["response"])
    if c_price.get("data") and c_price["data"].get("summaries"):
        for s in c_price["data"]["summaries"]:
            print(f" - {s['name']} | {s['price']} {s['currency']} | {'متوفر' if s['in_stock'] else 'غير متوفر'}")

    # Price range query
    c_price_range = chatbot_service.process_message(user_id="tester", message="اريد كريم سعره بين ال 10 وال 50")
    print("Chatbot response for 'اريد كريم سعره بين ال 10 وال 50':")
    print(c_price_range["response"])
    if c_price_range.get("data") and c_price_range["data"].get("summaries"):
        for s in c_price_range["data"]["summaries"]:
            print(f" - {s['name']} | {s['price']} {s['currency']} | {'متوفر' if s['in_stock'] else 'غير متوفر'}")

    print("Done. Inspect outputs above to verify behavior.")


if __name__ == '__main__':
    run_tests()
