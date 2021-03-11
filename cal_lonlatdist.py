# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:53:38 2020

@author: David
"""

import numpy as np
import pandas as pd

#计算两个经纬度之间的距离
def radian(degree):
    """
    把经纬度转换成弧度
    """
    return (degree*3.14159265)/180.0

def lon_lat_distance(lon1, lat1, lon2, lat2):
    #定义地球半径
    R = 6371
    
    #把角度转换成弧度
    radlon1 = radian(lon1)
    radlat1 = radian(lat1)
    radlon2 = radian(lon2)
    radlat2 = radian(lat2)
    
    dist = round(np.arccos(np.cos(radlat1) * np.cos(radlat2) * np.cos(radlon1-radlon2) + np.sin(radlat1) * np.sin(radlat2)) * R, 3)
    return dist
    
data = [1,2,3,4,5,6,7,8,9,10]

def mode(data):
    
    df = pd.DataFrame(data, columns=['data'])
    df['data'].mode().tolist()[0]
    
    data_unique = list(set(data))
    data_dict = dict.fromkeys(data_unique, 0)
    for i in data:
        data_dict[i] += 1
    data_series = pd.Series(data_dict)
    max_num = data_series.max()
    mode = data_series[data_series==max_num].index.tolist()[0]\
    
    return mode
    

[(('sh','bj'),) + ((2,3),)]

# 当众数不唯一时，会返回所有众数
df2 = pd.DataFrame([1,2,3], columns=['num'])
type(df2.num.mode())
df2.num.mean()

df3 = pd.DataFrame([[118.156, 39.639, 118.155, 39.638],\
                    [118.156, 39.639, 118.155, 39.64],\
                    [118.156, 39.639, 118.156, 39.638],\
                    [118.156, 39.639, 118.155, 39.639],\
                    [103.75, 27.32, 103.76, 27.33],\
                    [103.75, 27.32, 103.76, 27.32],\
                    [103.75, 27.32, 103.75, 27.33]],\
                    columns=['lg1', 'lat1', 'lg2', 'lat2'])

if __name__ == '__main__':
    lon_lat_distance(118.156, 39.639, 118.155, 39.638)  #0.14
    lon_lat_distance(118.156, 39.639, 118.155, 39.64) #0.14
    lon_lat_distance(118.156, 39.639, 118.156, 39.638) #0.111
    lon_lat_distance(118.156, 39.639, 118.155, 39.639) #0.086
    
    
    # 查看经纬度变换0.01，距离变化量
    lon_lat_distance(103.75, 27.32, 103.76, 27.33) #同时变，1.487
    lon_lat_distance(103.75, 27.32, 103.76, 27.32) #变经度：0.988
    lon_lat_distance(103.75, 27.32, 103.75, 27.33) #变纬度：1.112
    
    lon_lat_distance(df3.lg1, df3.lat1, df3.lg2, df3.lat2)
    
    lon_lat_distance(120.552, 32.359, 120.552, 32.359)
    






    