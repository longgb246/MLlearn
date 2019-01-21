# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/17
"""  
Usage Of '32_longest_valid_parentheses.py' : 
"""


class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        len_s = len(s)
        p = [0] * len_s
        max_len = 0
        for i in range(len_s - 1)[::-1]:
            if s[i] == '(' and (i + 1 + p[i + 1]) < len_s and s[i + 1 + p[i + 1]] == ')':
                p[i] = p[i + 1] + 2
                if i + 1 + p[i + 1] + 1 < len_s:
                    p[i] += p[i + 1 + p[i + 1] + 1]
                max_len = max(max_len, p[i])
        return max_len


# public int longestValidParentheses(String s) {
# 	char[] chars = s.toCharArray();
# 	return Math.max(calc(chars, 0, 1, chars.length, '('), calc(chars, chars.length -1, -1, -1, ')'));
# }
#
# private static int calc(char[] chars , int i ,  int flag,int end, char cTem){
# 	int max = 0, sum = 0, currLen = 0,validLen = 0;
# 	for (;i != end; i += flag) {
# 		sum += (chars[i] == cTem ? 1 : -1);
#           currLen ++;
# 		if(sum < 0){
# 			max = max > validLen ? max : validLen;
# 			sum = 0;
# 			currLen = 0;
#             validLen = 0;
# 		}else if(sum == 0){
#             validLen = currLen;
#       }
# 	}
# 	return max > validLen ? max : validLen;
# }


def get_test_instance(example=1):
    s = "(()"
    if example == 1:
        pass
    if example == 2:
        s = ")()())"
    if example == 3:
        s = ""
    if example == 4:
        s = "()(())"
    return s


def main():
    s = get_test_instance(example=1)
    # s = get_test_instance(example=2)
    # s = get_test_instance(example=3)
    # s = get_test_instance(example=4)
    # s = get_test_instance(example=5)
    res = Solution().longestValidParentheses(s)
    print(res)


if __name__ == '__main__':
    main()
