from services.mongo_service import mongo_service


def test_sha3ri_mutqasaf_variations():
    variants = [
        "شعري متقصف",
        "هل عندكم علاج لشعري متقصف؟",
        "شعري متقصف جدا ولا اعرف ماذا افعل"
    ]
    for q in variants:
        res = mongo_service.search_products(q, limit=5)
        assert isinstance(res, list)
        assert len(res) > 0, f"Expected at least one result for query '{q}'"
        assert res[0].get('product_id') == 'DUMMY7', f"Query '{q}' did not return DUMMY7 top; got {[r.get('product_id') for r in res[:3]]}"


def test_bashra_jafa_variations():
    variants = [
        "بشرة جافة",
        "بشرتي جافة جدا هل يوجد مرطب مناسب",
        "ابحث عن شيء لبشرة جافة"
    ]
    for q in variants:
        res = mongo_service.search_products(q, limit=5)
        assert isinstance(res, list)
        assert len(res) > 0, f"Expected at least one result for query '{q}'"
        assert res[0].get('product_id') == 'DUMMY10', f"Query '{q}' did not return DUMMY10 top; got {[r.get('product_id') for r in res[:3]]}"


if __name__ == '__main__':
    test_sha3ri_mutqasaf_variations()
    test_bashra_jafa_variations()
    print('OK')