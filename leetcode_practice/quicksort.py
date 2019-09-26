def quick_sort(x):

    def find_pivot(x, left, right):
        return x[int((left + right)/2)]

    def q_sort(x, left, right):
        if (left == right):
            return x
        pivot_value = find_pivot(x, left, right)
        partition_point = partition(x, left, right, pivot_value)
        q_sort(x, left, partition_point-1)
        q_sort(x, partition_point, right)
 
    def partition(x, left, right, pivot_value):
        while (left < right):
            if (x[left] < pivot_value):
                left += 1
            if (x[right] > pivot_value):
                right += -1
            if (x[left] >= pivot_value and x[right] <= pivot_value):
                tmp = x[left]
                x[left]=x[right]
                x[right]=tmp
                left += 1
                right += -1

        return left

    q_sort(x, 0, len(x)-1)

    return x

print(quick_sort([1,2,7,9,11,18,12,4,7,4,6,3,5,3,8]))