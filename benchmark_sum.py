import timeit

def sum_gen():
    return sum(1 for x in range(34) if x >= 2)

def len_list():
    return len([x for x in range(34) if x >= 2])

print("sum_gen:", timeit.timeit(sum_gen, number=100000))
print("len_list:", timeit.timeit(len_list, number=100000))
