from services.mongo_service import mongo_service
import re

q = "بشرتي جافة جدا هل يوجد مرطب مناسب"
print('Query:', q)
# copy of tokenization and scoring steps (lightweight)
def _simplify_ar(s: str) -> str:
    if not s:
        return ""
    v = s.lower()
    mappings = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': '', 'ڤ': 'ف'}
    for a, b in mappings.items():
        v = v.replace(a, b)
    v = re.sub(r'[^0-9\u0600-\u06FF\s]', '', v)
    v = re.sub(r'(^|\s)ال', r"\1", v)
    v = re.sub(r'\s+', ' ', v).strip()
    return v

def _normalize_token(t: str) -> str:
    t = t.strip()
    t = re.sub(r"^(?:my|ل)?","", t)
    t = re.sub(r"(ي|ها|نا|كم|هم)$", "", t)
    return t

raw_tokens = [t for t in re.split(r'\s+', q) if t.strip()]
tokens = [_simplify_ar(_normalize_token(t)) for t in raw_tokens if t.strip()]
print('Tokens:', tokens)

# token regex
token_regex = '|'.join([re.escape(tok) for tok in set(tokens) if tok])
print('Token regex:', token_regex)

cand_cursor = mongo_service.products.find({
    "$or": [
        {"title_ar": {"$regex": token_regex, "$options": "i"}},
        {"description_ar": {"$regex": token_regex, "$options": "i"}}
    ]
}, {"_id":0, "product_id":1, 'title_ar':1, 'description_ar':1}).limit(400)

cands = list(cand_cursor)
print('Candidates count:', len(cands))

symptom_map = {
    'متقصف': {'tokens': ['متقصف'], 'cats': ['العناية بالشعر', 'شعر']},
    'جاف': {'tokens': ['جاف', 'جافة'], 'cats': ['العناية بالبشرة', 'بشرة']},
    'تساقط': {'tokens': ['تساقط', 'تساقط الشعر'], 'cats': ['العناية بالشعر', 'شعر']},
}

scored = []
for p in cands:
    s_title = _simplify_ar(p.get('title_ar') or '')
    s_desc = _simplify_ar(p.get('description_ar') or '')
    match_count = 0
    exact_count = 0
    desc_bonus = 0
    s_title_words = set(s_title.split())
    s_desc_words = set(s_desc.split())
    for tok in tokens:
        if not tok:
            continue
        if tok in s_title or tok in s_desc:
            match_count += 1
        if tok in s_title_words or tok in s_desc_words:
            exact_count += 1
        if tok in s_desc_words:
            desc_bonus += 1
    # Compute detailed scoring breakdown (same as service)
    base = exact_count * 6 + match_count * 2 + desc_bonus * 2
    phrase_bonus = 10 if all((tok in s_title or tok in s_desc) for tok in tokens if tok) else 0
    symptom_boost = 0
    domain_boost = 0

    for sym, info in symptom_map.items():
        if any(sym_token in tok for tok in tokens for sym_token in info['tokens']):
            if any(sym_token in s_title or sym_token in s_desc for sym_token in info['tokens']):
                symptom_boost = 12
            else:
                cat = (p.get('category_ar') or p.get('category') or '').lower()
                attrs = p.get('attributes') or {}
                hair_type = [str(x).lower() for x in (attrs.get('hair_type') or [])]
                skin_type = [str(x).lower() for x in (attrs.get('skin_type') or [])]
                if any(c.lower() in cat for c in info['cats']) or any(any(sym_token in a for sym_token in info['tokens']) for a in hair_type + skin_type):
                    symptom_boost = 6
            break

    # detect domain from tokens (skin/hair)
    domain = None
    domain_map = {
        'skin': ['بشرت', 'بشرة', 'بشرتي', 'بشرتك', 'وجه', 'جلد'],
        'hair': ['شعر', 'شعري', 'شعرك']
    }
    for t in tokens:
        for dname, kws in domain_map.items():
            if any(kw in t for kw in kws):
                domain = dname
                break
        if domain:
            break

    if domain == 'skin' and ('بشر' in s_title or 'بشر' in s_desc or 'بشرة' in (p.get('category_ar') or '') or 'بشرة' in (p.get('category') or '')):
        domain_boost = 8
    if domain == 'hair' and ('شعر' in s_title or 'شعر' in s_desc or 'شعر' in (p.get('category_ar') or '') or 'شعر' in (p.get('category') or '')):
        domain_boost = 8

    score = base + phrase_bonus + symptom_boost + domain_boost

    scored.append((score, base, phrase_bonus, symptom_boost, domain_boost, exact_count, match_count, p))

scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
print('Top scored candidates:')
for t in scored[:10]:
    s, base, phrase, sympt, dom, ec, mc, p = t
    s_title = _simplify_ar(p.get('title_ar') or '')
    s_desc = _simplify_ar(p.get('description_ar') or '')
    print('score', s, 'base', base, 'phrase', phrase, 'symptom', sympt, 'domain', dom, 'exact', ec, 'match', mc, p.get('product_id'), repr(p.get('title_ar')))
    print('   s_title:', repr(s_title))
    print('   s_desc :', repr(s_desc))
