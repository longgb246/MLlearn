# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/23
"""  
Usage Of '76_min_window.py' : 
"""


class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 双指针 - 61.60%
        map_t = {}
        map_d = {}
        # 记录目标字符串每个字母出现次数
        for i in t:
            map_t[i] = map_t.get(i, 0) + 1
        start, begin, found = 0, -1, 0
        end = min_length = len_s = len(s)
        len_t = len(t)
        for i in range(len_s):
            map_d[s[i]] = map_d.get(s[i], 0) + 1  # 每来一个字符给它的出现次数加1
            if map_d[s[i]] <= map_t.get(s[i], 0):  # 如果加1后这个字符的数量不超过目标串中该字符的数量，则找到了一个匹配字符
                found += 1
            if found == len_t:  # 如果找到的匹配字符数等于目标串长度，说明找到了一个符合要求的子串
                # 将开头没用的都跳过，没用是指该字符出现次数超过了目标串中出现的次数，并把它们出现次数都减1
                while (start < i) and (map_d[s[start]] > map_t.get(s[start], 0)):
                    map_d[s[start]] = map_d[s[start]] - 1
                    start += 1
                if i - start < min_length:  # 这时候start指向该子串开头的字母，判断该子串长度
                    min_length = i - start
                    begin = start
                    end = i
                # 把开头的这个匹配字符跳过，并将匹配字符数减1
                map_d[s[start]] -= 1
                found -= 1
                start += 1
        res = '' if begin == -1 else s[begin:(end + 1)]
        return res

    def minWindow2(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 99.20%
        if len(t) > len(s):
            return ""
        from collections import defaultdict
        # d 用于记录还差 key 满足全包含的个数，-1表示有多的
        d = defaultdict(int)
        for i in t:
            d[i] += 1

        m, l, ml, cnt = 2 ** 30, 0, 0, 0
        # 正向遍历字符串
        for r in range(len(s)):
            c = s[r]
            if c in d:
                cnt = cnt + 1 if d[c] > 0 else cnt  # 找到的是>0(还差的)即+1，找到的事<=0(有多余的)不改变计数
                d[c] -= 1  # d中凡找到即递减，负数表示多余的
            while cnt == len(t):  # 找齐了
                if r - l < m:  # 小于最小长度
                    m = r - l  # 更新最小长度
                    ml = l  # 更新左边界
                c2 = s[l]
                if c2 in d:  # 当前左边界的值如果在t集合里
                    if d[c2] >= 0:  # 且该值的统计>=0，表示取出该值后，全包含就不满足，则cnt要-1
                        cnt -= 1
                    d[c2] += 1  # d统计值减少
                l += 1  # 依次移动左边界
        return '' if m == 2 ** 30 else s[ml:ml + m + 1]

    def circleMinWindow(self, s, t):
        s = s * 2
        res = self.minWindow2(s, t)
        return res


def get_test_instance(example=1):
    s = "ADOBECODEBANC"
    t = "ABC"
    if example == 1:
        pass
    if example == 2:
        s = "intention"
        t = "execution"  # 5
    return s, t


def main():
    s, t = get_test_instance(example=1)
    # s, t = get_test_instance(example=2)
    # res = Solution().minWindow(s, t)
    res = Solution().minWindow2(s, t)
    print(res)
    # circle
    Solution().circleMinWindow("ABOBECODEBANC", t)


if __name__ == '__main__':
    main()
