import pytest

from services.chatbot_service import ChatbotService


@pytest.fixture
def svc():
    return ChatbotService()


def make_product(pid, price=10.0, in_stock=True, rating=4.0, review_count=0):
    return {
        "product_id": pid,
        "name": f"Product {pid}",
        "price": price,
        "in_stock": in_stock,
        "rating": rating,
        "review_count": review_count
    }


def test_prefer_in_stock_over_out_of_stock(svc):
    p1 = make_product('p1', price=20.0, in_stock=True, rating=3.0, review_count=5)
    p2 = make_product('p2', price=20.0, in_stock=False, rating=5.0, review_count=100)
    chosen = svc._choose_preferred_product([p2, p1])
    assert chosen['product_id'] == 'p1'
    assert chosen['in_stock'] is True


def test_prefer_higher_rating_then_review_count(svc):
    p1 = make_product('p1', price=30.0, in_stock=True, rating=4.5, review_count=20)
    p2 = make_product('p2', price=30.0, in_stock=True, rating=4.7, review_count=5)
    chosen = svc._choose_preferred_product([p1, p2])
    assert chosen['product_id'] == 'p2'


def test_fallback_to_first_when_all_equal(svc):
    p1 = make_product('p1', price=15.0, in_stock=True, rating=4.0, review_count=10)
    p2 = make_product('p2', price=15.0, in_stock=True, rating=4.0, review_count=10)
    chosen = svc._choose_preferred_product([p1, p2])
    assert chosen['product_id'] == 'p1'
