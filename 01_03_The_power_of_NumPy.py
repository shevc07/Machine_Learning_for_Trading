import numpy as np
import time


def p06_creating_numpy_arrays():
    np1 = np.empty(5)
    np2 = np.empty((3, 4))
    np3 = np.empty((3, 4, 5))
    print(np1, np2, np3)


def p07_arrays_with_initial_numbers():
    np1 = np.ones((3,4), dtype=np.int)
    print(np1)
    np0 = np.zeros((4,5))
    print(np0)


def p09_generating_random_numbers():
    # random_sample [0.0, 1.0)
    np1 = np.random.random_sample((3,4))
    print(np1)
    np2 = np.random.rand(3, 4)
    print(np2)
    np3 = np.random.normal(size=(3,4))
    print(np3)
    np4 = np.random.normal(50, 10, size=(3,4))
    print(np4)
    np5 = np.random.randint(0, 10, (3,4))
    print(np5)


def p10_array_attributes():
    np1 = np.random.rand(4, 5, 2)
    print(np1.shape, np1.size, np1.dtype, np1.ndim)


def p11_operations_on_ndarrays():
    np.random.seed(777)
    np1 = np.random.randint(0,10,(4,5))
    print(np1)
    print("sum of array: {}".format(np1.sum()))
    print("sum by col: {}".format(np1.sum(axis=0)))
    print("sum by row: {}".format(np1.sum(axis=1)))
    print("min of each col: {}".format(np1.min(axis=0)))
    print("min of each row: {}".format(np1.min(axis=1)))
    print("max of each col: {}".format(np1.max(axis=0)))
    print("max of each row: {}".format(np1.max(axis=1)))
    print("mean of array: {}".format(np1.mean()))


def p12_locate_maximum_value():
    np.random.seed(777)
    np1 = np.random.randint(0,100,10)
    print("array: {}".format(np1))
    print("max of array: {}, location is {}".format(np1.max(), np1.argmax()))


def how_long(func, *args):
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return result, t1-t0


def manual_mean(arr):
    sum = 0
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            sum += arr[i, j]
    return sum / arr.size


def numpy_mean(arr):
    return arr.mean()


def p14_how_fast_is_numpy():
    np1 = np.random.rand(1000, 10000)
    r_manual, t_manual = how_long(manual_mean, np1)
    r_numpy, t_numpy = how_long(numpy_mean, np1)
    print("Result: manual:{}, numpy:{}, equal:{}".format(r_manual,r_numpy,True if abs(r_manual - r_numpy) <= 10e-6 else False))
    print("Time: manual:{}, numpy:{}, numpy {}x faster thean manual".format(t_manual,t_numpy,int(t_manual/t_numpy)))

def p15_accessing_array_elements():
    np1 = np.random.randint(0, 10, (4,5))
    print(np1)
    print(np1[:, 0:5:3])


def p17_indexing_an_array_with_another_array():
    np1 = np.random.randint(0,10,10)
    print(np1)
    print(np1[[1,1,7,7,4]])


def p18_boolean_or_mask_index_arrays():
    np1 = np.array([
        (12,4,56,7,45,34,2),
        (34,5,56,3,34,21,2)
    ])

    print(np1)
    print(np1[np1<np1.mean()])
    np1[np1<np1.mean()] = np1.mean()
    print(np1)


def test_run():
    p06_creating_numpy_arrays()
    p07_arrays_with_initial_numbers()
    p09_generating_random_numbers()
    p10_array_attributes()
    p11_operations_on_ndarrays()
    p12_locate_maximum_value()
    # p14_how_fast_is_numpy()
    p15_accessing_array_elements()
    p17_indexing_an_array_with_another_array()
    p18_boolean_or_mask_index_arrays()

if __name__ == "__main__":
    test_run()