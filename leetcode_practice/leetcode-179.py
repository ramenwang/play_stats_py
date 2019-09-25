def compare(x,y):
    return x+y > y+x

class compareStr(str):
    def __lt__(self,b):
        print('obj created with ',b)
        print('self created with ',self)
        is_lt = self+b > b+self
        print(is_lt)
        print("\n")
        return is_lt

class compareSelf(str):
    def __lt__(self,other):
        print('obj created with ',other)
        print('self created with ',self)
        print(self + other)
        return self+other

nums = [43,23,63,98]
# print(''.join(sorted(map(str,nums), key=largerNumber)))
str_list = map(str,nums)
print(sorted(str_list, key=compareStr))

