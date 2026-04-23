import timeit

def bench_gen():
    mentsu = [('koutsu', 1), ('shuntsu', 2), ('koutsu', 3), ('shuntsu', 4)]
    return sum(1 for m in mentsu if m[0] == 'koutsu')

def bench_list_comp():
    mentsu = [('koutsu', 1), ('shuntsu', 2), ('koutsu', 3), ('shuntsu', 4)]
    return len([m for m in mentsu if m[0] == 'koutsu'])

print("sum(gen):", timeit.timeit(bench_gen, number=1000000))
print("len([comp]):", timeit.timeit(bench_list_comp, number=1000000))
