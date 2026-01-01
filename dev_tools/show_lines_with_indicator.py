with open('data/assets/data.js','r',encoding='utf-8') as f:
    found = False
    for i, line in enumerate(f, 1):
        if 'title_ar' in line or '\u0600' <= line[:1] <= '\u06FF':
            print(i, line.rstrip())
            found = True
    if not found:
        print('No lines with title_ar or starting Arabic characters found')
