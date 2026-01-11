import sys, os
# Ensure project root is on sys.path when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
import json
from app import app

client = TestClient(app)

r = client.get('/test')
print('GET /test', r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))

payload = {"user_id":"tester","message":"أريد شراء زيت"}
r = client.post('/chat', json=payload)
print('\nPOST /chat', r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
