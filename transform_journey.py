# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:37:41 2020

@author: David
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time


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
    
    dist = round(np.arccos(np.cos(radlat1) * np.cos(radlat2) * np.cos(radlon1-radlon2) + np.sin(radlat1) * np.sin(radlat2)) * R, 5)
    return dist

#np.arccos(round(1.000000001,4))
#radlon1 = radian(120.07931599999999)
#radlat1 = radian(31.996765999999997)
#radlon2 = radian(120.07931599999999)
#radlat2 = radian(31.996765999999997)


#df_deviceid_selected = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求\数据分析建模-张俊/deviceid/deviceid_selected.csv', index_col=0)
#deviceid = df_deviceid_selected.loc[100, 'deviceid']

df = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/vin_sample/part-0.csv')
df = df.sort_values(by=['uploadtime'])
df.reset_index(drop=True, inplace=True)


df['datetime'] = (df['uploadtime']/1000).apply(datetime.fromtimestamp)
df['datetime_diff'] = df['datetime'].diff()

df['lg_next'] = df['lg'].shift(-1)
df['lat_next'] = df['lat'].shift(-1)
df = df.iloc[:-1, :]

distance = []
steps = df.shape[0]
for i in df.index:
    lon1, lat1, lon2, lat2 = df.loc[i, 'lg'], df.loc[i, 'lat'], df.loc[i, 'lg_next'], df.loc[i, 'lat_next']
    dist = lon_lat_distance(lon1, lat1, lon2, lat2)
    distance.append(dist)
    print('step:{}/{}  distance:{}'.format(i, steps, dist))

df.loc[:, 'distance'] = distance

df_null = df[df['distance'].isnull()]


#analysis
df_3 = df[df['datetime_diff'].dt.days<0]
vc = df['operationmode'].value_counts()


df['vehiclespeed'].hist(bins=20)
plt.title('vehiclespeed')


df_cs = df[df['vehiclespeed']!=0.0]
df_cs['vehiclespeed'].hist(bins=40)
plt.title('vehiclespeed')

df_cs['datetime'].dt.hour.hist(bins=30)
plt.xticks(np.arange(12)*2)
plt.title('driving time')

from datetime import date
df_cs.loc[:, 'weekday'] = df_cs['datetime'].dt.date.apply(date.isoweekday)
#datetime.strptime("2019-03-04","%Y-%m-%d").isoweekday()

df_cs_w = df_cs[df_cs['weekday']<6]
df_cs_w['weekday'].hist(bins=30)
plt.title('driving working day')

df_cs_h = df_cs[df_cs['weekday']>=6]
df_cs_h['weekday'].hist(bins=30)
plt.title('driving weekend')



df_cs_w['datetime'].dt.hour.hist(bins=30)
plt.xticks(np.arange(12)*2)
plt.title('driving time')


df_cs_h['datetime'].dt.hour.hist(bins=30)
plt.xticks(np.arange(12)*2)
plt.title('driving time')

df_1 = df[df['operationmode']=='INVALID']



#停车位置分析
#vehiclestatus：STOPPED
#chargingstatus：CHARGING_STOPPED，停车充电
vc = df['chargingstatus'].value_counts()
vc.plot.bar()

plt.scatter(df['lg'], df['lat'])

df_1 = df[df['lg']<=0]
vc = df_1['vehiclestatus'].value_counts()
vc = df_1['operationmode'].value_counts()
vc = df_1['chargingstatus'].value_counts()
vc = df_1['dt'].value_counts()

df_2 = df[df['lg']>0]
plt.scatter(df_2['lg'], df_2['lat'])
plt.xticks(np.arange(12)/4 + 118)

df_location = df[df['chargingstatus']=='CHARGING_STOPPED']
df_location.reset_index(drop=True, inplace=True)

df_location = df_location.sort_values(by=['uploadtime'])
df_location.reset_index(drop=True, inplace=True)

df_location['lg'].astype(int).value_counts()

df_1 = df_location[df_location['lg']<=0]

df_2 = df_location[df_location['lg']>0]
plt.scatter(df_2['lg'], df_2['lat'])


#急加速
df_accpedtrav = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/df_accpedtrav/part-0.csv')
dc = df_accpedtrav.describe().round(2)
vc = df_accpedtrav['accpedtrav'].value_counts()
df_accpedtrav['accpedtrav'].hist(bins=60)
plt.xlabel('accpedtrav')
plt.ylabel('count')
plt.show()
#选择占比千分之一以上的数据
accpedtrav = vc[vc>=100].index.tolist()
df_accpedtrav_select = df_accpedtrav[df_accpedtrav['accpedtrav'].isin(accpedtrav)]
df_accpedtrav_select.reset_index(drop=True, inplace=True)

dc_select = df_accpedtrav_select.describe().round(2)
vc_select = df_accpedtrav_select['accpedtrav'].value_counts()
df_accpedtrav_select['accpedtrav'].hist(bins=60)
plt.xlabel('accpedtrav')
plt.ylabel('count')
plt.show()

#急刹车
df_brakepedstat = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/df_brakepedstat/part-0.csv')
dc = df_brakepedstat.describe().round(2)
vc = df_brakepedstat['brakepedstat'].value_counts()
df_brakepedstat['brakepedstat'].hist(bins=40)
plt.xlabel('brakepedstat')
plt.ylabel('count')
plt.show()
#选择占比千分之一以上的数据
brakepedstat = vc[vc>=100].index.tolist()
df_brakepedstat_select = df_brakepedstat[df_brakepedstat['brakepedstat'].isin(brakepedstat)]
df_brakepedstat_select.reset_index(drop=True, inplace=True)

dc_select = df_brakepedstat_select.describe().round(2)
vc_select = df_brakepedstat_select['brakepedstat'].value_counts()
df_brakepedstat_select['brakepedstat'].hist(bins=40)
plt.xlabel('brakepedstat')
plt.ylabel('count')
plt.show()


#转换成行程
#条件：前后时间相隔超过5分钟；单次出行里程不足1km不记作出行次数。
def to_journey(df):
    #读取数据
    df = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/df_vin/part-0.csv')
    df_select = df[(df['vehiclestatus']=='STARTED') & (df['operationmode']=='EV') & (df['chargingstatus']=='NO_CHARGING')]
    df_select.reset_index(drop=True, inplace=True)
        
    #清理异常经纬度
#    df_select = df_select[df_select['lg']>0]
#    df_select.reset_index(drop=True, inplace=True)
#    
#    df1 = df_select[df_select['lg']<=0]
    
    #对时间排序
    df_select = df_select.sort_values(by=['uploadtime'])
    df_select.reset_index(drop=True, inplace=True)
    
    #计算相隔两次上传时间的时间插
    df_select['datetime'] = (df_select['uploadtime']/1000).apply(datetime.fromtimestamp)
    df_select['datetime_diff'] = df_select['datetime'].diff()
    
    #判断行程终止点
    df_select.loc[:, 'journey_end'] = df_select['datetime_diff'].dt.total_seconds()>300
    df_select.loc[:, 'journey_end'] = df_select['journey_end'].map({True:1, False:0})
    
    #判断是否是行程
    df_select.loc[:, 'is_journey'] = np.nan
    df_select.loc[:, 'journey_status'] = np.nan
    df_select.loc[:, 'journey_id'] = np.nan
    start_index = 0
    journey_id = 0
    for i in df_select.index:
        status = df_select.loc[i, 'journey_end']
        if status==1:
            end_index = i-1
            #创建journey_id
            for j in np.arange(start_index, end_index+1):
                df_select.loc[j, 'journey_id'] = journey_id
            journey_id += 1
            distance = df_select.loc[end_index, 'accmiles'] - df_select.loc[start_index, 'accmiles']
            if distance<1:
                for k in np.arange(start_index, end_index+1):
                    df_select.loc[k, 'is_journey'] = 0
            df_select.loc[start_index, 'journey_status'] = 'START'
            df_select.loc[end_index, 'journey_status'] = 'END'
            print('step:', i, ' status:', status, ' start_index:', start_index, 'end_index:', end_index)
            start_index = end_index + 1
    
    if end_index==df_select.shape[0]-2:
        df_select = df_select.iloc[:-1, :]
    else:
        df_select.loc[start_index, 'journey_status'] = 'START'
        df_select.loc[df_select.shape[0]-1, 'journey_status'] = 'END'
        for i in np.arange(start_index, df_select.shape[0]):
            df_select.loc[i, 'journey_id'] = journey_id
        
    df_select.fillna(value={'is_journey': 1}, inplace=True)
    df_select.fillna(value={'journey_status': ''}, inplace=True)
    
    df_select = df_select[df_select['is_journey']==1]
    df_select.reset_index(drop=True, inplace=True)
    
    #转换成行程
    start_time = []
    end_time = []
    duration = []
    distance = []
    max_speed = []
    min_speed = []
    avg_speed = []
    soc_consumption =[]
    stop_times = []
    rush_speed_times = []
    nasty_brake_times = []
    rush_times_std = []
    nasty_times_std = []
    stop_times_std = []
    soc_std = []
    city = []
    highway = []
#    journey_list = df_select['journey_id'].unique().astype(int)
    steps = df_select['journey_id'].unique().shape[0]
    deviceid = [df_select.loc[0, 'deviceid']] * steps
    i = 1
    for journey_id in df_select['journey_id'].unique().astype(int):
        df_tmp = df_select[df_select['journey_id']==journey_id]
        df_tmp.reset_index(drop=True, inplace=True)
        print('step:{}/{} records_num:{}'.format(i, steps, df_tmp.shape[0]))
        i += 1
        
        start_time.append(df_tmp.loc[0, 'datetime'])
        end_time.append(df_tmp.loc[df_tmp.shape[0]-1, 'datetime'])
        duration.append(df_tmp.loc[df_tmp.shape[0]-1, 'datetime'] - df_tmp.loc[0, 'datetime'])
        distance.append(round(df_tmp.loc[df_tmp.shape[0]-1, 'accmiles'] - df_tmp.loc[0, 'accmiles'],2))
        max_speed.append(df_tmp['vehiclespeed'].max())
        min_speed.append(df_tmp['vehiclespeed'].min())
        avg_speed.append(df_tmp['vehiclespeed'].mean())
        soc_consumption.append(df_tmp.loc[0, 'soc'] - df_tmp.loc[df_tmp.shape[0]-1, 'soc'])
        stop_times.append(df_tmp[df_tmp['vehiclespeed']==0.0].shape[0])
        rush_speed_times.append(df_tmp[df_tmp['accpedtrav']>24.0].shape[0])
        nasty_brake_times.append(df_tmp[df_tmp['brakepedstat']>17.0].shape[0])
        rush_times_std.append(rush_speed_times[-1] * 100 / distance[-1])
        nasty_times_std.append(nasty_brake_times[-1] * 100 / distance[-1])
        stop_times_std.append(stop_times[-1] * 100 / distance[-1])
        soc_std.append(soc_consumption[-1] * 100 / distance[-1])
        city.append(((df_tmp[df_tmp['vehiclespeed']<80.0].shape[0] / df_tmp.shape[0]) > 0.5) * 1)
        highway.append(1 - city[-1])

    df_result = pd.DataFrame({"deviceid":deviceid, "start_time":start_time, "end_time":end_time, "duration":duration,\
                              "distance":distance, "max_speed":max_speed, "min_speed":min_speed, "avg_speed":avg_speed,\
                              "soc_consumption":soc_consumption, "stop_times":stop_times, "rush_speed_times":rush_speed_times,\
                              "nasty_brake_times":nasty_brake_times, "rush_times_std":rush_times_std, "nasty_times_std":nasty_times_std,\
                              "stop_times_std":stop_times_std, "soc_std":soc_std, "city":city, "highway":highway})

    
    #行程持续时间分布
    duration_dist = df_result['duration'].dt.seconds / 60
    duration_dist.hist(bins=50)
    plt.xlabel('duration')
    plt.ylabel('counts')
    plt.title('journey duration')
    
    #行程距离分布
    df_result['distance'].hist(bins=50)
    plt.xlabel('distance')
    plt.ylabel('counts')
    plt.title('journey distance')
    
    #行程平均速度分布
    df_result['avg_speed'].hist(bins=50)
    plt.xlabel('avg_speed')
    plt.ylabel('counts')
    plt.title('journey avg_speed')
    
    #行程耗电量分布
    df_result['soc_consumption'].hist(bins=50)
    plt.xlabel('soc_consumpiton')
    plt.ylabel('counts')
    plt.title('journey soc_consumpiton')
    
    #行程耗停车次数分布
    df_result['stop_times'].hist(bins=50)
    plt.xlabel('stop_times')
    plt.ylabel('counts')
    plt.title('journey stop_times')
  
    #行程急加速数分布
    df_result['rush_speed_times'].hist(bins=50)
    plt.xlabel('rush_speed_times')
    plt.ylabel('counts')
    plt.title('journey rush_speed_times')
    
    #行程急减速数分布
    df_result['nasty_brake_times'].hist(bins=50)
    plt.xlabel('nasty_brake_times')
    plt.ylabel('counts')
    plt.title('journey nasty_brake_times')
    
    #每百公里急加速次数分布
    df_result['rush_times_std'].hist(bins=50)
    plt.xlabel('rush_times_std')
    plt.ylabel('counts')
    plt.title('rush speed times per 100KM')
    
    #每百公里急减速次数分布
    df_result['nasty_times_std'].hist(bins=50)
    plt.xlabel('nasty_times_std')
    plt.ylabel('counts')
    plt.title('nasty brake times per 100KM')
    
    #每百公里停车次数分布
    df_tmp = df_result[df_result['stop_times_std']<=250]
    df_tmp['stop_times_std'].hist(bins=50)
    plt.xlabel('stop_times_std')
    plt.ylabel('counts')
    plt.title('stop times per 100KM')
    
    #每百公里耗电量分布
    df_result['soc_std'].hist(bins=50)
    plt.xlabel('soc_consumption_std')
    plt.ylabel('counts')
    plt.title('soc_consumption per 100KM')
    
    #市内和高速分布
    df_result['city'].sum()
    plt.xlabel('soc_consumption_std')
    plt.ylabel('counts')
    plt.title('soc_consumption per 100KM')
    






    
