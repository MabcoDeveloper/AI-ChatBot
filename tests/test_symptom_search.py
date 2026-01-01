# test_symptom_search.py
"""Unit test: symptom-style Arabic query 'شعري متقصف' should return the Protein Repair product (DUMMY7) as the top result.
"""
from services.mongo_service import mongo_service


def test_sha3ri_mutqasaf_returns_dummy7():
    q = "شعري متقصف"
    res = mongo_service.search_products(q, limit=10)
    assert isinstance(res, list)
    assert len(res) > 0, f"Expected at least one result for query '{q}'"
    top = res[0]
    # The canonical product id for Protein Repair in assets is expected to be 'DUMMY7'
    assert top.get('product_id') == 'DUMMY7', f"Top result product_id was {top.get('product_id')} (expected 'DUMMY7')"


if __name__ == '__main__':
    # For manual runs
    test_sha3ri_mutqasaf_returns_dummy7()
    print("OK")
