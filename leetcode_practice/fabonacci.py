def fibonacci(n):
    mem = [0]*(n+1)
    def recur(n, mem):
        if n==1:
            mem[n] = 1
        elif n>1:
            mem[n] = fibonacci(n-2)+fibonacci(n-1)
        return mem[n]

    return recur(n, mem)
    

n = int(input())
print(fibonacci(n))