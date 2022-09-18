import numpy as np

# a = np.array([[1],[2],[3],[4]])
# print(a[1])

# a_append = np.append(a, [5])
# print(a_append[4])
#
# a_delete = np.delete(a, 1)
# print(a_delete)
#
#
# print(len(a))

""""""
#
# a = np.array([1,2,3,4])
# print(a)
#
# a[0] = 100
#
#
# print(a)
# #
#
# empty = np.array([[], [], [], []])
#
# print(f"empty.shape = {empty.shape}")
# new = np.array([[1], [2], [3], [4]])
# print(f"new.shape = {new.shape}")
#
# result = np.append(empty, new, axis=1)
# print(result)
# new2 = np.array([[5], [6], [7], [8]])
#
# result2 = np.append(result, new, axis=1)
# result3 = np.append(result2, new2, axis=1)
# print(result2)
# print(f"result3 = {result3}")
#
# mean = np.mean(result3, axis=1)
# print(mean)

### Delete
# result3 = np.delete(result3, 0, axis=1)
# print(f"result3 (deleted) = {result3}")
# result2 = np.delete(result3, 0, axis=1)
# print(f"result2 (deleted) = {result2}")

#
# new = np.array([1,2,3,4])
# new = new.reshape(4,1)
# print(new)
#


""""""
# arr = np.array([], dtype=np.uint8)
#
# for i in range(5):
#     arr = np.append(arr, i)
#     print(arr)


""""""

range_ratio1 = 7.206015288829803e-05
max,  min = 0.00843080971390009, 0.008358749561011791
range = max - min

print(f"{range_ratio1:.6f} == {range:.6f}")

val_nmzd = (range_ratio1-min)/range
print(f"{range_ratio1} - {min} / {range} = {(range_ratio1-min)/range}  == {val_nmzd}")

# ratios-min/ratio[1] = (0.00843-0.0083600003272295)/7.000000186963007e-05 = 0.99996
# normalized_ratio[1] = 0.9999571083350962
