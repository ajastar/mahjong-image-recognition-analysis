import timeit

setup = """
hand_34 = [0, 0, 1, 2, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
"""

s1 = "sum(1 for c in hand_34 if c >= 2)"
s2 = "len([c for c in hand_34 if c >= 2])"

t1 = timeit.timeit(s1, setup=setup, number=100000)
t2 = timeit.timeit(s2, setup=setup, number=100000)

print(f"sum(): {t1:.4f}s")
print(f"len(): {t2:.4f}s")
