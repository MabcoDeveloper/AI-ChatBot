from services.mongo_service import mongo_service
from models.intent_classifier import IntentClassifier

def gather_examples(limit=10000):
    sessions = mongo_service.fetch_training_sessions({}, limit=limit)
    texts, labels = [], []
    for s in sessions:
        for t in s.get('turns', []):
            if t.get('role') == 'user':
                txt = t.get('normalized') or t.get('message')
                intent = t.get('intent')
                if txt and intent:
                    texts.append(txt)
                    labels.append(intent)
    return texts, labels

if __name__ == '__main__':
    texts, labels = gather_examples()
    print(f"Found {len(texts)} labeled examples")
    if len(texts) < 10:
        print("Not enough data to train. Collect more conversations first.")
    else:
        clf = IntentClassifier()
        report = clf.train(texts, labels)
        clf.save('models/intent_model.joblib')
        print('Model trained and saved to models/intent_model.joblib')
        print('Summary:', report)
