from collections import Counter
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        track = []
        def backtrack(nums, track):
            count = Counter(track)
            if count[")"] > count["("]:
                return
            if count[")"] > nums or count["("] > nums:
                return
            if len(track) == nums*2:
                res.append(''.join(track.copy()))
                return

            for i in ["(", ")"]:
                track.append(i)
                backtrack(nums, track)
                track.pop()
        backtrack(n, track)
        return res

s = Solution()
print(s.generateParenthesis(3))