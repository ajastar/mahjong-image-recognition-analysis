import sys
from unittest.mock import MagicMock
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.jit'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['numba'] = MagicMock()
import mahjong_logic
import server
import single_bot
import Jantama_Attach_Mode

print("Mahjong logic imported ok")
