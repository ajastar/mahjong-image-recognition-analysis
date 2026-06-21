import sys

class MockModule:
    pass

sys.modules['numpy'] = MockModule()
sys.modules['cv2'] = MockModule()
sys.modules['numba'] = MockModule()
sys.modules['torch'] = MockModule()
sys.modules['torch.nn'] = MockModule()
sys.modules['ultralytics'] = MockModule()
sys.modules['pyautogui'] = MockModule()
sys.modules['win32gui'] = MockModule()
sys.modules['win32ui'] = MockModule()
sys.modules['win32con'] = MockModule()
sys.modules['win32api'] = MockModule()

import Background_Multi_Mahjong
import Jantama_Attach_Mode
import factory_main
import single_bot

print("All imports successful!")
