# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/2/24
"""  
Usage Of 'quick_sort.py' : 
"""

from __future__ import print_function


def quick_sort(array):
    if len(array) < 2: return array
    return quick_sort([lt for lt in array[1:] if lt < array[0]]) \
           + [array[0]] \
           + quick_sort([ge for ge in array[1:] if ge >= array[0]])


def main():
    iList = [3, 14, 2, 12, 9, 9, 33, 99, 35]
    print(quick_sort(iList))


if __name__ == '__main__':
    main()
