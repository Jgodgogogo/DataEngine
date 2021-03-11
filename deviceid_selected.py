# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:11:05 2020

@author: David
"""

import numpy as np
import pandas as pd


m1 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_1/part-1.csv')
m2 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_2/part-2.csv')
m3 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_3/part-3.csv')
m4 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_4/part-4.csv')
m5 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_5/part-5.csv')
m6 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_6/part-6.csv')
m7 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_7/part-7.csv')
m8 = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/month_8/part-8.csv')


selected = []
num = m1.shape[0]
for i in np.arange(num):
    did = m1.loc[i,'deviceid']
    if (did in m2['deviceid'].tolist()) and (did in m3['deviceid'].tolist()) and (did in m4['deviceid'].tolist()\
        and (did in m5['deviceid'].tolist()) and (did in m6['deviceid'].tolist()) and (did in m7['deviceid'].tolist())\
        and (did in m8['deviceid'].tolist())):
        
        selected.append(did)
        print('{}/{}  {}'.format(i, num, did))
        

df_selelct = pd.DataFrame(selected, columns=['deviceid'])
df_selelct.to_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid_selected.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
