from typing import List, Dict, Any, Optional
import os
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    # scikit-learn (and joblib) not available in the environment. We set symbols to None and
    # a flag so callers can raise informative errors rather than failing on import.
    Pipeline = None
    TfidfVectorizer = None
    LogisticRegression = None
    train_test_split = None
    classification_report = None
    joblib = None
    SKLEARN_AVAILABLE = False

class IntentClassifier:
    """Simple sklearn-based intent classifier with TF-IDF + LogisticRegression"""

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None

    def build_pipeline(self):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed. Install with: pip install scikit-learn joblib")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=20000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])

    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed. Install with: pip install scikit-learn joblib")
        if not self.pipeline:
            self.build_pipeline()
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=42, stratify=labels)
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        if verbose:
            print(classification_report(y_test, preds, zero_division=0))
        return report

    def predict(self, text: str) -> str:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed. Install with: pip install scikit-learn joblib")
        if not self.pipeline:
            raise RuntimeError('Model not trained or loaded')
        return self.pipeline.predict([text])[0]

    def predict_proba(self, text: str) -> Dict[str, float]:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed. Install with: pip install scikit-learn joblib")
        if not self.pipeline:
            raise RuntimeError('Model not trained or loaded')
        classes = self.pipeline.classes_
        probs = self.pipeline.predict_proba([text])[0]
        return dict(zip(classes, probs.tolist()))

    def save(self, path: str):
        if joblib is None:
            raise RuntimeError("joblib is not available. Install scikit-learn and joblib to save models.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str):
        if joblib is None:
            raise RuntimeError("joblib is not available. Install scikit-learn and joblib to load models.")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.pipeline = joblib.load(path)
