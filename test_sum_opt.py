import timeit

setup_code = """
hand_34 = [0] * 34
hand_34[0] = 2; hand_34[1] = 2; hand_34[2] = 2; hand_34[3] = 2
hand_34[4] = 2; hand_34[5] = 2; hand_34[6] = 2

mentsu = [("koutsu", 1), ("shunta", 2), ("koutsu", 3), ("koutsu", 4), ("koutsu", 5)]
blocks = [{'type': 'koutsu'}, {'type': 'shuntsu'}, {'type': 'koutsu'}, {'type': 'koutsu'}]
suit_counts = [3, 0, 8]
"""

tests = {
    "terminals (orig)": "sum(hand_34[t] for t in [0,8,9,17,18,26] + list(range(27,34)))",
    "terminals (opt)": "hand_34[0]+hand_34[8]+hand_34[9]+hand_34[17]+hand_34[18]+hand_34[26]+hand_34[27]+hand_34[28]+hand_34[29]+hand_34[30]+hand_34[31]+hand_34[32]+hand_34[33]",

    "suit_nonzero (orig)": "sum(1 for x in suit_counts if x > 0)",
    "suit_nonzero (opt)": "(suit_counts[0]>0) + (suit_counts[1]>0) + (suit_counts[2]>0)",

    "koutsu (orig)": "sum(1 for m in mentsu if m[0] == 'koutsu')",
    "koutsu (opt2)": "len([1 for m in mentsu if m[0] == 'koutsu'])",
    "koutsu (opt3)": "c=0\nfor m in mentsu:\n  if m[0] == 'koutsu': c+=1",

    "blocks_koutsu (orig)": "sum(1 for b in blocks if b['type'] == 'koutsu')",
    "blocks_koutsu (opt)": "c=0\nfor b in blocks:\n  if b['type'] == 'koutsu': c+=1",

    "pairs (orig)": "sum(1 for c in hand_34 if c >= 2)",
    "pairs (opt)": "c=0\nfor x in hand_34:\n  if x >= 2: c+=1",
}

for name, code in tests.items():
    t = timeit.timeit(code, setup=setup_code, number=100000)
    print(f"{name:20}: {t:.5f}")
