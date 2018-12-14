# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2018/12/13
"""  
Usage Of 'tets_get_from_hive' : 
"""

import os
import sys

# python test_get_from_hive.py.py 2018-11-28

this_date = sys.argv[1]
date_str = '_'.join(this_date.split('-')[1:])

train_sql = '''
hive -e 'set hive.cli.print.header=true; select * from app.two_step_train_df_sql_lgb_dev_t' > train_df_{date_str}.tsv
'''.format(date_str=date_str)

target_sql = '''
hive -e 'set hive.cli.print.header=true; select * from app.two_step_target_df_lgb_dev_t' > target_df_{date_str}.tsv
'''.format(date_str=date_str)

preprocess_sql = '''
hive -e 'set hive.cli.print.header=true; select sku_id, store_id, sale, is_longtail, dt, Granu_split from app.two_step_preprocess_df_lgb_dev_t order by sku_id, store_id, dt' > preprocess_df_{date_str}.tsv
'''.format(date_str=date_str)

pre_data_sql = '''
hive -e 'set hive.cli.print.header=true; select sku_code, store_id, sale_list, sale_type from app.dev_lgb_test_rst_dev_all_t where tenant_id=28 and dt="{this_date}" ' > pre_data_{date_str}.tsv
'''.format(this_date=this_date, date_str=date_str)

print(preprocess_sql)
os.system(preprocess_sql)

print(pre_data_sql)
os.system(pre_data_sql)

''' 
hive -e 'set hive.cli.print.header=true; select sku_code, store_id, sale, sale_date from app.app_saas_sfs_model_input_pre where tenant_id = 28 and dt = "ACTIVE" '  > input_data.tsv
'''

tar_exec = ''' tar -zcvf {date_str}_data.tar.gz *.tsv '''.format(date_str=date_str)
print(tar_exec)
os.system(tar_exec)

rm_tsv_exec = ''' rm *.tsv '''
print('Clean the *.tsv ')
os.system(rm_tsv_exec)
