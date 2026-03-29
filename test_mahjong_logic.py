import mahjong_logic

def test_check_calls_priority():
    hand_34 = [0]*34
    hand_34[0] = 2 # Pair of 1m
    hand_34[1] = 1 # 2m
    hand_34[2] = 1 # 3m
    hand_34[3] = 1 # 4m
    hand_34[4] = 1 # 5m
    hand_34[5] = 1 # 6m
    # To not be agari, we need 13 tiles that don't win.
    hand_34[9] = 1 # 1p
    hand_34[10] = 1 # 2p
    hand_34[11] = 1 # 3p
    hand_34[18] = 3 # 1s
    hand_34[27] = 1 # 1z

    # target_tile is 1m (idx 0), from_offset=3 (left)
    # Give it a dora of 1m (idx 0) so the PON score goes > 0
    # Expect CHI and PON to be available, but PON has priority
    result = mahjong_logic.NakiUtils.check_calls(hand_34, 0, 3, [0], 10)

    print(f"Result: {result}")
    assert result == "PON (0)"
    print("Test passed.")

if __name__ == "__main__":
    test_check_calls_priority()
