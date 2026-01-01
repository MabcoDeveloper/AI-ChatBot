import re
text=open('data/assets/data.js','r',encoding='utf-8').read()
for m in re.finditer(r'export\s+const\s+DummyProducts', text):
    print('found at', m.start())
print('Total occurrences:', len(list(re.finditer(r'export\s+const\s+DummyProducts', text))))
