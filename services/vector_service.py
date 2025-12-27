from typing import List, Dict, Any

class VectorService:
    """Minimal vector service placeholder for product embeddings and search.

    This lightweight implementation provides the methods the app expects:
    - load_products_from_mongo(mongo_service): loads products into an in-memory list
    - products_data: list of product dicts
    - find_similar(query, top_n=5): naive similarity based on token overlap
    """

    def __init__(self):
        self.products_data: List[Dict[str, Any]] = []

    def load_products_from_mongo(self, mongo_service) -> int:
        """Load products from the provided `mongo_service` into memory.

        Returns the number of products loaded. Silently falls back to an empty
        index if Mongo is unavailable or an error occurs.
        """
        try:
            cursor = getattr(mongo_service, "products").find({})
            self.products_data = []
            for p in cursor:
                self.products_data.append({
                    "id": str(p.get("_id")),
                    "title": p.get("title", ""),
                    "description": p.get("description", ""),
                    "price": p.get("price"),
                    "category": p.get("category"),
                })
            return len(self.products_data)
        except Exception:
            self.products_data = []
            return 0

    def find_similar(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top_n products by simple token overlap with the query."""
        q_tokens = set(query.split())
        scored = []
        for doc in self.products_data:
            title_tokens = set(str(doc.get("title", "")).split())
            score = len(q_tokens & title_tokens)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_n]]


# Export singleton instance expected by the rest of the app
vector_service = VectorService()
