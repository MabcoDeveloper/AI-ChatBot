import sys, os, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.chatbot_service import chatbot_service

try:
    res = chatbot_service.process_message('tester', 'أريد شراء زيت')
    print('Result:', res)
except Exception as e:
    print('Exception raised:')
    traceback.print_exc()
