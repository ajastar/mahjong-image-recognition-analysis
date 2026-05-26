import timeit

setup_code = """
hand_34 = [0] * 34
hand_34[0] = 2; hand_34[1] = 2; hand_34[2] = 2; hand_34[3] = 2
hand_34[4] = 2; hand_34[5] = 2; hand_34[6] = 2
"""

tests = {
    "pairs (orig)": "sum(1 for c in hand_34 if c >= 2)",
    "pairs (opt_loop)": "c=0\nfor x in hand_34:\n  if x >= 2: c+=1",
    "pairs (opt_list)": "len([1 for x in hand_34 if x >= 2])",
}

for name, code in tests.items():
    t = timeit.timeit(code, setup=setup_code, number=100000)
    print(f"{name:20}: {t:.5f}")
