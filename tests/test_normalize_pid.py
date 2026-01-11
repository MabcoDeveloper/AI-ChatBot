import sys, os
# Ensure project root is on sys.path so tests can import application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from services.chatbot_service import ChatbotService

try:
    from bson.objectid import ObjectId
    HAS_OBJECTID = True
except Exception:
    HAS_OBJECTID = False


@pytest.fixture
def svc():
    return ChatbotService()


def test_dict_with_oid(svc):
    inp = {'_id': {'$oid': '695f7251a09e737894a02b94'}}
    assert svc._normalize_pid(inp) == '695f7251a09e737894a02b94'


@pytest.mark.skipif(not HAS_OBJECTID, reason="bson.ObjectId not available")
def test_objectid_instance(svc):
    oid = ObjectId('6963f2ca6e2d8355841a1c60')
    # when passed as an ObjectId instance
    assert svc._normalize_pid(oid) == str(oid)
    # and when embedded under _id
    assert svc._normalize_pid({'_id': oid}) == str(oid)


def test_raw_string_id(svc):
    assert svc._normalize_pid('CUST_SERUM1') == 'CUST_SERUM1'


def test_product_dict_with_product_id_only(svc):
    assert svc._normalize_pid({'product_id': 'CUST_SERUM1'}) == 'CUST_SERUM1'


# extra: handle ObjectId-like repr strings
def test_objectid_repr_string(svc):
    s = "ObjectId('6963f2ca6e2d8355841a1c60')"
    assert svc._normalize_pid(s) == '6963f2ca6e2d8355841a1c60'