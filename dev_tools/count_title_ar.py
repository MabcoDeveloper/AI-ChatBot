text=open('data/assets/data.js','r',encoding='utf-8').read()
print('title_ar occurrences in file:', text.count('title_ar'))
print('first 400 chars of file:\n', text[:400])
