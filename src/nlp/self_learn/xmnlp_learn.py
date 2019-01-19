# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2019/1/15
"""  
Usage Of 'xmnlp_learn.py' : 
"""

import xmnlp

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

for each in xmnlp.seg(doc, hmm=True):
    print(each)

postag_v = None
hmm_v = None

# 408 + 111 + 179 + 92 + 116 + 340
# 1246
# 4-tag(BMES)和6-tag(BB2B3MES)
