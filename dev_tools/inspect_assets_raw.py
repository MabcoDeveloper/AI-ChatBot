import os, re, json
assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'assets', 'data.js')
text = open(assets_path, 'r', encoding='utf-8').read()
arr_match = re.search(r'export\s+const\s+DummyProducts\s*=\s*(\[.*?\])\s*;', text, flags=re.S)
if not arr_match:
    print('No DummyProducts found')
    raise SystemExit(1)
arr_text = arr_match.group(1)
try:
    products = json.loads(arr_text)
    print('JSON loads succeeded, count', len(products))
except Exception as e:
    print('JSON loads failed:', e)
    t = arr_text.replace("'", '"')
    t = re.sub(r'([\{\[],)\s*([a-zA-Z0-9_\-]+)\s*:', r'\1 "\2":', t)
    t = re.sub(r',\s*([\]\}])', r'\1', t)
    products = json.loads(t)
    print('Fallback parsing succeeded, count', len(products))

# find entry with _id '3'
try:
    for p in products:
        if str(p.get('_id')) == '3' or p.get('title','').startswith('Keratin'):
            print('Found raw asset item _id=3')
            print('keys:', list(p.keys()))
            print('title_ar raw repr:', repr(p.get('title_ar')))
            print('description_ar raw repr:', repr(p.get('description_ar')))
            break
    else:
        print('Product with _id 3 not found in parsed products')
except Exception as e:
    print('Could not iterate parsed products:', e)
    # fallback: print raw text around the _id occurrence
    m = re.search(r"\{[^\}]{0,400}['_\"]?_id['_\"]?\s*:\s*\"3\"[^\}]{0,400}\}", text, flags=re.S)
    if m:
        print('Raw snippet around _id 3:')
        print(m.group(0))
    else:
        # try a simpler nearby search
        idx = text.find('\n  {\n    _id: "3"')
        if idx>=0:
            print('Raw nearby lines:')
            print(text[idx:idx+300])
        else:
            print('Could not find raw _id 3 snippet')
