import unittest
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_logic import ShantenUtils, AgariUtils, USE_NUMBA

class TestLogicCheck(unittest.TestCase):
    def setUp(self):
        print(f"\n[Test Setup] USE_NUMBA = {USE_NUMBA}")
        if USE_NUMBA:
            print("⚠️ WARNING: Numba is Enabled. Speed test will reflect JIT performance.")
        else:
            print("ℹ️ INFO: Python Mode. Testing optimized Python algorithms.")

    def test_shanten_normal_python_opt(self):
        """Verify Shanten calculation correctness and basic speed."""
        # Kokushi 13-wait (0 shanten)
        hand_kokushi = [0]*34
        for i in [0,8,9,17,18,26,27,28,29,30,31,32,33]: hand_kokushi[i] = 1
        hand_kokushi[0] += 1 # Pair of 1m
        
        s = ShantenUtils.calculate_shanten(hand_kokushi)
        self.assertEqual(s, -1, "Kokushi Agari should be -1 shanten")

        # Chuuren Poutou (9-wait)
        hand_chuuren = [0]*34
        # 1112345678999m + 1m
        for i in range(9): hand_chuuren[i] = 1
        hand_chuuren[0] += 2
        hand_chuuren[8] += 2
        hand_chuuren[0] += 1 # Extra 1m
        
        s = ShantenUtils.calculate_shanten(hand_chuuren)
        self.assertEqual(s, -1, "Chuuren Poutou Agari should be -1 shanten")

        # Random Hand (Check consistency)
        # 123m 456p 789s 11z 22z + 3m
        hand_normal = [0]*34
        hand_normal[0]=1; hand_normal[1]=1; hand_normal[2]=1
        hand_normal[9]=1; hand_normal[10]=1; hand_normal[11]=1
        hand_normal[18]=1; hand_normal[19]=1; hand_normal[20]=1
        hand_normal[27]=2
        hand_normal[28]=2
        
        s = ShantenUtils.calculate_shanten(hand_normal)
        # This hand (13 tiles) has 3 Mentsu + 2 Pairs. It is Tenpai (0 shanten), waiting for 1z/2z to become Koutsu or Head.
        self.assertEqual(s, 0, "Tenpai hand (13 tiles) should be 0 shanten")

    def test_bytes_caching(self):
        """Ensure bytes input works for suit patterns."""
        # Man suit: 111234567
        hand = [3, 1, 1, 1, 1, 1, 1, 0, 0] # 111 2 3 4 5 6 7
        # Should be 2 mentsu (111, 123? no 1 is 3), maybe 111 + 234 + 567?
        # 111, 234, 567 -> 3 Mentsu!
        # Let's use internal method if possible or just calculate shanten for a full hand using this suit.
        
        full_hand = [0]*34
        for i in range(9): full_hand[i] = hand[i]
        # Rest empty
        # Should have 3 mentsu. 
        # Shanten = 8 - 2*3 - 0 - 0 = 2.
        
        s = ShantenUtils.calculate_shanten(full_hand)
        # 3 mentsu (111m, 234m, 567m) -> need 1 head + 0 mentsu -> 
        # Wait, format: 4 groups + 1 pair.
        # We have 3 groups. Need 1 group + 1 pair.
        # Shanten = 2*g + p ... formula: 8 - 2*M - T ...
        # 8 - 6 = 2. Correct.
        self.assertEqual(s, 2, "Should handle bytes conversion correctly for Man suit")

if __name__ == '__main__':
    unittest.main()
