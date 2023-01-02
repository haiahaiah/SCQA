
import numpy as np
from scipy import stats


def get_p_value(arrA, arrB):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a,b)
    return p


if __name__ == "__main__":
    nums1 = [73.6754, 73.2234, 73.2987]
    nums2 = [74.2847, 74.004, 73.8191]
    print(sum(nums1) / len(nums1))
    print(sum(nums2) / len(nums2))
    print(get_p_value(nums1, nums2))
