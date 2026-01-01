import re
text=open('data/assets/data.js','r',encoding='utf-8').read()
ids=[m.group(1) for m in re.finditer(r"\b_id\s*:\s*\"(\d+)\"", text)]
print('Found', len(ids), 'ids')
for sid in ids:
    idx=text.find(f'_id: "{sid}"')
    window=text[idx:idx+400]
    print('\n---- window for id', sid, '----')
    print(window)
    m_title=re.search(r'title_ar\s*:\s*"([^\"]*)"', window)
    m_desc=re.search(r'description_ar\s*:\s*"([^\"]*)"', window)
    print(sid, 'title_ar:', m_title.group(1) if m_title else None, 'desc_ar:', (m_desc.group(1)[:40] + '...') if m_desc else None)
