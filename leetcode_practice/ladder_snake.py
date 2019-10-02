# find the least step to 

class Solution(object):
    def snakesAndLadders(self, board):
        """
        :type board: List[List[int]]
        :rtype: int
        """
        
        N = len(board)
        
        def get_rc(num):
            quotient, remainder = divmod(num-1, N)
            row = N-quotient-1
            col = remainder if row % 2 == 1 else N - remainder - 1
            return row, col
        
        dist = {1:0}
        queue = [1]

        while queue:
            num = queue.pop(0)
            if num == N**2: return dist[num]
            for iNum in range(num+1, min(num+6, N**2)+1):
                row, col = get_rc(iNum)
                if board[row][col] != -1:
                    iNum = board[row][col]
                if iNum not in dist:
                    dist[iNum] = dist[num] + 1
                    queue.append(iNum)
        
        return -1

