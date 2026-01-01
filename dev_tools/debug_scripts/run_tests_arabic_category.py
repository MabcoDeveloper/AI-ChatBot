import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import importlib

mod = importlib.import_module('tests.test_arabic_category_search')

if __name__ == '__main__':
    failures = []
    for name in dir(mod):
        if name.startswith('test_'):
            func = getattr(mod, name)
            try:
                func()
                print(f"{name}: OK")
            except AssertionError as e:
                print(f"{name}: FAIL -> {e}")
                failures.append((name, e))
            except Exception as e:
                print(f"{name}: ERROR -> {e}")
                failures.append((name, e))
    if failures:
        print('\nSome tests failed')
        sys.exit(1)
    print('\nAll tests passed')
