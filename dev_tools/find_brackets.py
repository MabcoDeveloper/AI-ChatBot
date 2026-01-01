import re
text = open('data/assets/data.js','r',encoding='utf-8').read()
for idx, m in enumerate(re.finditer(r'\];', text)):
    print(idx, '->', m.start(), '... context ...')
    print(text[m.start()-40:m.start()+40])
