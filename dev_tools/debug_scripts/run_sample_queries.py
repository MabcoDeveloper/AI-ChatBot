import os
import sys
# Ensure project root is on sys.path so 'services' package imports correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from services.chatbot_service import chatbot_service

samples = [
    ("user1", "مرحبا"),
    ("user2", "عندكم شامبو للشعر الجاف؟"),
    ("user3", "بكم أحمر شفاه ماك؟"),
    ("user4", "هل عندكم عروض؟"),
    ("user5", "أريد أفضل شامبو"),
    ("user6", "أريد شراء أحمر شفاه ماك"),
    ("user6", "أحمد, 0501234567"),
    ("user7", "كم سعر بين 30 و 100"),
    ("user8", "انهي الجلسة"),
    ("user9", "هل يوجد سيروم فيتامين سي متوفر؟")
]

for uid, msg in samples:
    try:
        res = chatbot_service.process_message(uid, msg)
    except Exception as e:
        print("-"*60)
        print(f"User: {uid} | Msg: {msg}")
        print("Exception while processing:", e)
        continue
    print("-"*60)
    print(f"User: {uid} | Msg: {msg}")
    # result may occasionally be a (str, None) tuple for errors; handle both
    if isinstance(res, tuple):
        print("Result (tuple):", res)
        continue
    print(f"Intent: {res.get('intent')} (conf={res.get('intent_confidence')})")
    print(f"Response:\n{res.get('response')}\n")
    if res.get('data'):
        print("Data:", res.get('data'))
    print(f"Context summary: {res.get('context_summary')}\n")
