# -*- coding:utf-8 -*-
"""
  Author  : 'longguangbin'
  Contact : lgb453476610@163.com
  Date    : 2018/10/15
  Usage   :
"""

import pandas as pd
from fbprophet import Prophet

m = Prophet()
m.fit(df)

