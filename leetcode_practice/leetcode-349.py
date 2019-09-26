class Solution(object):
    
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        def unique(x):
            uni = []
            for iX in x:
                if not iX in uni:
                    uni.append(iX)
            return uni
            
        return unique([i for i in nums1 if i in nums2])

    def intersection2(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        set1 = set(nums1)
        set2 = set(nums2)
        
        if len(set1) < len(set2):
            return [i for i in set1 if i in set2]
        else:
            return [i for i in set2 if i in set1]
