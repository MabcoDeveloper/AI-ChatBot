import os, re, json

assets_path = os.path.join('data', 'assets', 'data.js')
text = open(assets_path, 'r', encoding='utf-8').read()
start_marker = 'export const DummyProducts'
start_idx = text.find(start_marker)
if start_idx < 0:
    print('No DummyProducts marker')
    raise SystemExit(1)
arr_start = text.find('[', start_idx)
arr_end = text.find('];', arr_start)
if arr_start < 0 or arr_end < 0:
    print('Could not find array bounds')
    raise SystemExit(1)
arr_text = text[arr_start:arr_end+1]
print('arr_text sample start (400 chars):')
print(arr_text[:400])
# show raw arr_text snippet around id 7
mraw = re.search(r"\{[^\}]*_id\s*:\s*\"7\"([^\}]*)\}", arr_text, flags=re.S)
if mraw:
    print('raw arr_text snippet for id 7:')
    print(mraw.group(0))
else:
    print('raw arr_text snippet for id 7 not found')
# search arr_text for title_ar
print('title_ar in original arr_text?', 'title_ar' in arr_text)
if 'title_ar' in arr_text:
    idx = arr_text.find('title_ar')
    print(arr_text[idx-80:idx+80])

arr_text = re.sub(r'//.*', '', arr_text)
arr_text = re.sub(r',\s*([\]\}])', r'\1', arr_text)
try:
    products = json.loads(arr_text)
except Exception as e:
    print('json.loads failed, trying fallback:', e)
    t = arr_text.replace("'", '"')
    t = re.sub(r'([\{\[,])\s*([a-zA-Z0-9_\-]+)\s*:', r'\1 "\2":', t)
    t = re.sub(r',\s*([\]\}])', r'\1', t)
    # debug: check if title_ar exists in t
    print('title_ar in transformed text?', 'title_ar' in t)
    print('Sample fragment around title_ar:')
    ti = t.find('title_ar')
    if ti>=0:
        print(t[ti-80:ti+80])
    products = json.loads(t)
print('Parsed count:', len(products))
# print the arr_text window around id 7 to see raw JS content
m2 = re.search(r'\{[^\}]*"?_id"?\s*:\s*"7"([^\}]*)\}', arr_text, flags=re.S)
if m2:
    print('JS window for id 7:')
    print(m2.group(0))
else:
    print('Could not find JS window for id 7 in arr_text')

# print product 7 raw
for p in products:
    if p.get('_id') == '7':
        print('raw asset for id 7:', p)
        break
# check presence of title_ar
missing = [p.get('_id') for p in products if not p.get('title_ar') or not p.get('description_ar')]
print('Missing count in raw parsed assets:', len(missing))
if missing:
    print('Missing ids sample:', missing[:10])
else:
    print('All parsed products have Arabic fields')
