# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:47:16 2020

@author: David
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import os

# 出行特征
df_vehicle_behavior = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_vehicle_behavior/dbh_vehicle_behavior.csv')
#df_vehicle_behavior.dtypes
#deviceid           object
#trip_id             int64
#start_time         object
#stop_time          object
#trip_distance     float64
#start_hour          int64
#start_soc           int64
#stop_soc            int64
#trip_duration     float64
#delta_soc           int64
#distance_range      int64
#soc_range           int64

df_charging_type =  pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_vihecle_type/dbh_vihecle_type.csv', dtype='str')
df1 = pd.merge(df_vehicle_behavior, df_charging_type, on=['deviceid'], how='left')
# 行程距离
df_vehicle_behavior['trip_distance'].hist(bins=50)
plt.title('trip_distance(km) distribution')
plt.xlabel('trip distance')
plt.ylabel('count')
df_vehicle_behavior['trip_distance'].describe()

df1 = df_vehicle_behavior[df_vehicle_behavior['trip_distance']<=20]
df1['trip_distance'].hist(bins=50)
plt.title('trip_distance(km) distribution')
plt.xlabel('trip distance')
plt.ylabel('count')
plt.xticks(np.arange(21))

# 行程时长
df_vehicle_behavior['trip_duration'].hist(bins=50)
plt.title('trip_duration(munites) distribution')
plt.xlabel('trip duration')
plt.ylabel('count')
df_vehicle_behavior['trip_duration'].describe()

df1 = df_vehicle_behavior[df_vehicle_behavior['trip_duration']<=40]
df1['trip_duration'].hist(bins=50)
plt.title('trip_duration distribution')
plt.xlabel('trip duration')
plt.ylabel('count')
plt.xticks(np.arange(20)*2)

# 出行时间分布
df_vehicle_behavior['start_hour'].hist(bins=50)
plt.title('start_hour distribution')
plt.xticks(np.arange(24))
plt.xlabel('start hour')
plt.ylabel('count')
df_vehicle_behavior[(df_vehicle_behavior['start_hour']>=0) & (df_vehicle_behavior['start_hour']<=6)].shape[0] / df_vehicle_behavior.shape[0]
df_vehicle_behavior[(df_vehicle_behavior['start_hour']>=7) & (df_vehicle_behavior['start_hour']<=12)].shape[0] / df_vehicle_behavior.shape[0]
df_vehicle_behavior[(df_vehicle_behavior['start_hour']>=13) & (df_vehicle_behavior['start_hour']<=18)].shape[0] / df_vehicle_behavior.shape[0]
df_vehicle_behavior[df_vehicle_behavior['start_hour']>=19].shape[0] / df_vehicle_behavior.shape[0]

# 出行SOC分布
df_vehicle_behavior['start_soc'].hist(bins=100)
plt.title('start_soc distribution')
plt.xlabel('start_soc')
plt.ylabel('count')
df_vehicle_behavior['start_soc'].describe()

# 出行Delta_SOC分布
df_vehicle_behavior['delta_soc'].hist(bins=100)
plt.title('delta_soc distribution')
plt.xlabel('delta_soc')
plt.ylabel('count')
df_vehicle_behavior['delta_soc'].describe()

df1 = df_vehicle_behavior[df_vehicle_behavior['delta_soc']<=10]
df1['delta_soc'].hist(bins=100)
plt.title('delta_soc distribution')
plt.xlabel('delta_soc')
plt.ylabel('count')
plt.xticks(np.arange(11))

# 行程距离与耗电量
plt.scatter(df_vehicle_behavior['trip_distance'], df_vehicle_behavior['delta_soc'])
plt.title('trip_distance VS delta_soc')
plt.xlabel('trip_distance')
plt.ylabel('delta_soc')
plt.xticks(np.arange(12)*20)
plt.yticks(np.arange(12)*20)

df_vehicle_behavior['power_consumption_rate'] = df_vehicle_behavior['delta_soc'] / df_vehicle_behavior['trip_distance']
df_vehicle_behavior['power_consumption_rate'].describe()
df1 = df_vehicle_behavior[df_vehicle_behavior['power_consumption_rate']>0]
df1['power_consumption_rate'].describe()
plt.scatter(df1['trip_distance'], df1['delta_soc'], c='b', alpha=0.2)
#plt.plot(df1['trip_distance'], df1['trip_distance']*0.43, color='r')
plt.title('trip_distance VS delta_soc')
plt.xlabel('trip_distance')
plt.ylabel('delta_soc')
plt.xticks(np.arange(12)*20)
plt.yticks(np.arange(12)*20)

# 求天数
df_vehicle_behavior['start_time'] = pd.to_datetime(df_vehicle_behavior['start_time'])
year, month=df_vehicle_behavior['start_time'][0].date().year, df_vehicle_behavior['start_time'][0].date().month
date1 = datetime.date(year=year, month=month, day=1)
if month<12:
    date2 = date1.replace(month=month+1)
else:
    date2 = date1.replace(year=year+1, month=1)
days = (date2-date1).days

# 行程数量/天
df2 = df1[['deviceid', 'trip_id']].groupby('deviceid').count()
df2.rename(columns={'trip_id':'trip_count'}, inplace=True)
df1 = pd.merge(df1, df2, how='inner', on='deviceid')
df2['trip_count'].hist(bins=100)

# 行程距离/天
df1 = df_vehicle_behavior[['deviceid', 'trip_distance']].groupby('deviceid').sum()/days
df1.rename(columns={'trip_distance':'trip_distance_day'}, inplace=True)
df_vehicle_behavior = pd.merge(df_vehicle_behavior, df1, how='inner', on='deviceid')
df1['trip_distance_day'].hist(bins=80)

# 行程数量 vs 行程距离
plt.scatter(df_vehicle_behavior['trip_count_day'], df_vehicle_behavior['trip_distance_day'])
plt.show()

# 行程时长/天
df1 = df_vehicle_behavior[['deviceid', 'trip_duration']].groupby('deviceid').sum()/days
df1.rename(columns={'trip_duration':'trip_duration_day'}, inplace=True)
df_vehicle_behavior = pd.merge(df_vehicle_behavior, df1, how='inner', on='deviceid')
df1.trip_duration_day.hist(bins=80)


del df_vehicle_behavior, df1
df_driving_behavior = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_driving_behavior/dbh_driving_behavior.csv')
df_driving_behavior.dtypes
#deviceid                 object
#trip_id                   int64
#start_time               object
#stop_time                object
#trip_distance           float64
#max_speed               float64
#avg_speed               float64
#std_speed               float64
#rush_times                int64
#nasty_times               int64
#rush_times_per100km     float64
#nasty_times_per100km    float64
#start_soc                 int64
#stop_soc                  int64
#delta_soc                 int64
#delta_soc_per100km      float64
#mxal_times                int64
#celohwn_times             int64
#lsocwn_times              int64
#start_hour                int64

# 行程最高速度分布
df_driving_behavior['max_speed'].hist(bins=100)
plt.title('max_speed distribution')
plt.xlabel('max speed')
plt.ylabel('count')
df_driving_behavior['max_speed'].describe()

# 行程平均速度分布
df_driving_behavior['avg_speed'].hist(bins=100)
plt.title('avg_speed distribution')
plt.xlabel('avg_speed')
plt.ylabel('count')
plt.xticks(np.arange(10)*10)
df_driving_behavior['avg_speed'].describe()

# 急加速分布
df_driving_behavior['rush_times'].hist(bins=100)
plt.title('rush_times distribution')
plt.xlabel('rush_times')
plt.ylabel('count')
df_driving_behavior['rush_times'].describe()
df1 = df_driving_behavior[df_driving_behavior['rush_times']<=20]
df1['rush_times'].hist(bins=100)
plt.title('rush_times distribution')
plt.xlabel('rush_times')
plt.ylabel('count')
plt.xticks(np.arange(11)*2)


# 急减速分布
df_driving_behavior['nasty_times'].hist(bins=100)
plt.title('nasty_times distribution')
plt.xlabel('nasty_times')
plt.ylabel('count')
df_driving_behavior['nasty_times'].describe()
df1 = df_driving_behavior[df_driving_behavior['nasty_times']<=10]
df1['nasty_times'].hist(bins=100)
plt.title('nasty_times distribution')
plt.xlabel('nasty_times')
plt.ylabel('count')
plt.xticks(np.arange(11))

# 每百公里急加速、每百公里急减速
df_driving_behavior['rush_times_per100km'].hist(bins=100)
plt.title('rush_times_per100km distribution')
plt.xlabel('rush_times_per100km')
plt.ylabel('count')
df_driving_behavior['rush_times_per100km'].describe()

df_driving_behavior['nasty_times_per100km'].hist(bins=100)
plt.title('nasty_times_per100km distribution')
plt.xlabel('nasty_times_per100km')
plt.ylabel('count')
df_driving_behavior['nasty_times_per100km'].describe()

#df1 = df_driving_behavior[(df_driving_behavior['rush_times_per100km']>0) & (df_driving_behavior['rush_times_per100km']<=100)]
#df2 = df_driving_behavior[(df_driving_behavior['nasty_times_per100km']>0) & (df_driving_behavior['nasty_times_per100km']<=100)]
df1 = df_driving_behavior[df_driving_behavior['rush_times_per100km']<=100]
df2 = df_driving_behavior[df_driving_behavior['nasty_times_per100km']<=100]
df1['rush_times_per100km'].hist(bins=60, alpha=0.4, facecolor='r', label='rush')
df2['nasty_times_per100km'].hist(bins=60, alpha=0.4, facecolor='b', label='nasty')
plt.title('rush_times_per100km VS nasty_times_per100km')
plt.xlabel('rush_nasty times')
plt.ylabel('count')
plt.legend()

df_driving_behavior['mxal_times'].sum()
df_driving_behavior['celohwn_times'].sum()
df_driving_behavior['lsocwn_times'].sum()



del df_driving_behavior, df1, df2
df_soc_rule = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_soc_rule/dbh_soc_rule.csv')
df_soc_rule.dtypes
#deviceid                 object
#start_time               object
#stop_time                object
#start_hour                int64
#trip_distance           float64
#avg_speed               float64
#std_speed               float64
#rush_times                int64
#nasty_times               int64
#rush_times_per100km     float64
#nasty_times_per100km    float64
#start_soc                 int64
#delta_soc                 int64
#delta_soc_per100km      float64

# 每百公里delta_soc分布
df_soc_rule['delta_soc_per100km'].hist(bins=100)
plt.title('delta_soc_per100km distribution')
plt.xlabel('delta_soc_per100km')
plt.ylabel('count')
df_soc_rule['delta_soc_per100km'].describe()

df1 = df_soc_rule[(df_soc_rule['delta_soc_per100km']>0) & (df_soc_rule['delta_soc_per100km']<=100)]
df1 = df1[(df1['rush_times_per100km']>0) & (df1['rush_times_per100km']<=100)]
df1 = df1[(df1['nasty_times_per100km']>0) & (df1['nasty_times_per100km']<=100)]
df1.reset_index(drop=True, inplace=True)

# 每百公里急加速次数分布和每百公里急减速次数分布
df1['rush_times_per100km'].hist(bins=100, alpha=0.6, label='rush')
df1['nasty_times_per100km'].hist(bins=100, alpha=0.6, label='nasty')
plt.title('rush_times_per100km and nasty_times_per100km ditribution')
plt.xlabel('rush_times_per100km and nasty_times_per100km')
plt.ylabel('count')
plt.legend()
df1['rush_times_per100km'].describe()
df1['nasty_times_per100km'].describe()



# 耗电率分布
df1['delta_soc_per100km'].hist(bins=100)
plt.title('delta_soc_per100km')
plt.xlabel('delta_soc_per100km')
plt.ylabel('count')
df1['delta_soc_per100km'].describe()

# 出行时间与每百公里耗电量
#f = {'delta_soc_per100km': ['median', 'std', 'quantile(0.25)', 'quantile(0.75)']}
df_25 = df1.groupby(['start_hour']).quantile(0.25)
df_50 = df1.groupby(['start_hour']).quantile(0.50)
df_75 = df1.groupby(['start_hour']).quantile(0.75)

plt.scatter(df1['start_hour'], df1['delta_soc_per100km'], alpha=0.1)
plt.plot(df_25.index, df_25['delta_soc_per100km'], color='g', label=0.25)
plt.plot(df_50.index, df_50['delta_soc_per100km'], color='r', label=0.50)
plt.plot(df_75.index, df_75['delta_soc_per100km'], color='y', label=0.75)
plt.title('start_hour vs delta_soc_per100km')
plt.xlabel('start_hour')
plt.ylabel('delta_soc')
plt.xticks(np.arange(25))
plt.legend()

#df_soc_rule_0 = df_soc_rule[(df_soc_rule['delta_soc_per100km']<200) & (df_soc_rule['delta_soc_per100km']!=0)]
#df_soc_rule_0.reset_index(drop=True, inplace=True)
#plt.scatter(df_soc_rule_0['start_hour'], df_soc_rule_0['delta_soc_per100km'])
#plt.title('start_hour and delta_soc_per100km')
#plt.xlabel('start_hour')
#plt.ylabel('delta_soc_per100km')

# 平均速度与耗电率
plt.scatter(df1['avg_speed'], df1['delta_soc_per100km'], alpha=0.2)
plt.title('avg_speed vs delta_soc_per100km')
plt.xlabel('avg_speed')
plt.ylabel('delta_soc_per100km')
df1['avg_speed'].corr(df1['delta_soc_per100km'])

# 速度标准差与耗电率
plt.scatter(df1['std_speed'], df1['delta_soc_per100km'], alpha=0.2)
plt.title('std_speed vs delta_soc_per100km')
plt.xlabel('std_speed')
plt.ylabel('delta_soc_per100km')
df1['std_speed'].corr(df1['delta_soc_per100km'])

# 百公里急加速次数与耗电率
plt.scatter(df1['rush_times_per100km'], df1['delta_soc_per100km'], alpha=0.2)
plt.title('rush_times_per100km vs delta_soc_per100km')
plt.xlabel('rush_times_per100km')
plt.ylabel('delta_soc_per100km')
df1['rush_times_per100km'].corr(df1['delta_soc_per100km'])

# 百公里急减速次数与耗电率
plt.scatter(df1['nasty_times_per100km'], df1['delta_soc_per100km'], alpha=0.2)
plt.title('nasty_times_per100km vs delta_soc_per100km')
plt.xlabel('nasty_times_per100km')
plt.ylabel('delta_soc_per100km')
df1['nasty_times_per100km'].corr(df1['delta_soc_per100km'])

# 起始电量与耗电率
plt.scatter(df1['start_soc'], df1['delta_soc_per100km'], alpha=0.3)
plt.title('start_soc and delta_soc_per100km')
plt.xlabel('start_soc')
plt.ylabel('delta_soc_per100km')
df1['start_soc'].corr(df1['delta_soc_per100km'])

# 起始电量范围与耗电率
df_25 = df1.groupby(['soc_range']).quantile(0.25)
df_50 = df1.groupby(['soc_range']).quantile(0.50)
df_75 = df1.groupby(['soc_range']).quantile(0.75)
plt.scatter(df1['soc_range'], df1['delta_soc_per100km'], alpha=0.3)
plt.plot(df_25.index, df_25['delta_soc_per100km'], color='orange', label=0.25)
plt.plot(df_50.index, df_50['delta_soc_per100km'], color='red', label=0.50)
plt.plot(df_75.index, df_75['delta_soc_per100km'], color='green', label=0.75)
plt.title('soc_range and delta_soc_per100km')
plt.xlabel('soc_range')
plt.ylabel('delta_soc_per100km')
plt.xticks(np.arange(11)*10)
plt.legend()
df1['soc_range'].corr(df1['delta_soc_per100km'])

#行程距离与耗电率
plt.scatter(df1['trip_distance'], df1['delta_soc_per100km'], alpha=0.1)
plt.title('trip_distance vs delta_soc_per100km')
plt.xlabel('trip_distance')
plt.ylabel('delta_soc_per100km')
df1['trip_distance'].corr(df1['delta_soc_per100km'])


# 功率
del df_soc_rule, df1, df_25, df_50, df_75
df = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_trip_basic/dbh_trip_basic.csv')
df1 = df[['vehiclespeed', 'accpedtrav', 'brakepedstat', 'power']]
df1['vehiclespeed'].corr(df1['power'])

df1['power'].hist(bins=100)
plt.title('power distribution')
plt.xlabel('power')
plt.ylabel('count')

df2 = df1[df1['power']!=0]
df2['power'].hist(bins=100)

df3 = df1[df1['power']>0]
df3['power'].describe()

df3['power'].hist(bins=100)
plt.title('power distribution')
plt.xlabel('power')
plt.ylabel('count')

df3['vehiclespeed'].corr(df3['power'])
df3['accpedtrav'].corr(df3['power'])
df3['brakepedstat'].corr(df3['power'])


# 充电行为
df_charging =  pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_charging/dbh_charging.csv', dtype='str')
df_charging['start_time'] = pd.to_datetime(df_charging['start_time'])

df_charging_type =  pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_vihecle_type/dbh_vihecle_type.csv', dtype='str')
df1 = pd.merge(df_charging, df_charging_type, how='inner', on=['deviceid'])


# 充电时长
df_charging['charging_duration'] = df_charging['charging_duration'].astype('float64')
df1 = df_charging['charging_duration']/60
plt.hist(df1, bins=100)
plt.title("charging duration distibution")
plt.xlabel("charging duration")
plt.ylabel("count")
plt.show()

#私家车，出租车
df1['charging_duration'] = df1['charging_duration'].astype(int)/60
df1['charging_frequency'] = df1['charging_frequency'].astype(int)
df1['vihecle_type'] = df1['vihecle_type'].astype(int)
color= df1['vihecle_type']
plt.scatter(df1['charging_frequency'], df1['charging_duration'], c=color, alpha=0.3)
plt.xlabel('charging_frequency')
plt.ylabel('charging_duration')
#plt.legend()
plt.show()





df1 = df_charging[df_charging['charging_duration']>0]
df1['charging_duration_2'] = df1['charging_duration']/60
plt.hist(df1['charging_duration']/60, bins=100)
plt.title("charging duration distibution (0,+)")
plt.xlabel("charging duration")
plt.ylabel("count")
plt.show()

df1['charging_duration_2'].describe()
df1['charging_duration_2'].quantile([0.8, 0.9, 0.95])

# 充电开始小时
df1['start_hour'] = df1['start_time'].dt.hour

df2 = df1[['deviceid', 'start_hour']].groupby('start_hour').count()
df2.rename(columns={'deviceid':'count'}, inplace=True)
plt.bar(df2.index, height=df2['count'])
plt.xlabel("hour")
plt.ylabel("count")
plt.title("charging times every hour")
plt.xticks(df2.index)
plt.show()

# 充电次数/月
df2 = df1[['deviceid', 'charging_id']].groupby('deviceid').count()
df2.rename(columns={'charging_id':'count'}, inplace=True)
plt.hist(df2['count'], bins=80)
plt.title('charging count per month disctribution')
plt.xlabel('charging count')
plt.ylabel('count')
plt.show()

df2['count'].describe()

# 充电起始SOC，终止SOC
df1['start_soc'].hist(bins=100)
plt.title('charging start soc disctribution')
plt.xlabel('charging start soc')
plt.ylabel('count')
plt.show()

df1['stop_soc'].hist(bins=100)
plt.title('charging stop soc disctribution')
plt.xlabel('charging stop soc')
plt.ylabel('count')
plt.show()

df1['stop_soc'].describe()

df2 = df1[df1['stop_soc']==0]


# 充电量
df1['delta_soc'].hist(bins=100)
plt.title('charging delta soc disctribution')
plt.xlabel('charging delta soc')
plt.ylabel('count')
plt.show()

# 充电效率（充电量/充电时长）
df1['charging_efficiency'] = df1['delta_soc']/(df1['charging_duration']/60)
df1['charging_efficiency'].hist(bins=100)
plt.title('charging efficiency disctribution')
plt.xlabel('charging efficiency')
plt.ylabel('count')
plt.show()

df1['charging_efficiency'].describe()

df2 = df1[df1['charging_efficiency']<150]
df2['charging_efficiency'].hist(bins=100)
plt.title('charging efficiency <150 disctribution')
plt.xlabel('charging efficiency')
plt.ylabel('count')
plt.show()

df2 = df1[df1['charging_efficiency']<=20]
len(df2['deviceid'].unique())

# 不同SOC充电效率
df2['charging_efficiency_greater_12.5'] = (df2['charging_efficiency']>12.5) * 1
df3 = df2[df2['charging_efficiency']>12.5]
from sklearn.linear_model import RidgeCV
x = np.array(df3['start_soc']).reshape(df3['start_soc'].shape[0], 1)
y = np.array(df3['charging_efficiency'])
model = RidgeCV()
model.fit(x,y)
pred = model.predict(np.arange(100).reshape(100,1))
model.get_params()
model.intercept_
model.coef_[0]


plt.scatter(df2['start_soc'], df2['charging_efficiency'], c=df2['charging_efficiency_greater_12.5'])
plt.plot(np.arange(100), pred, color='r')
plt.title('charging efficiency vs soc_range\n(charging efficiency<=20)')
plt.xlabel('start_soc')
plt.ylabel('charging efficiency')
plt.text(75, 16.5, 'y = {0}x + {1}'.format(round(model.coef_[0],4), round(model.intercept_,4)))
plt.show()





# 两次充电之间里程间隔(分私家车，出租车)
df4 = df_charging[(df_charging['delta_miles']>0) & (df_charging['delta_miles']<300)]
df4['delta_miles'].hist(bins=100)

# 判断家gps



#########################判断私家车、出租车、其他##################################

# 出租车，私家车 充电其实时间，充电时长
v1 = df_trip[['deviceid', 'vihecle_type_final']]
v2 = pd.merge(df_charging, v1, on='deviceid', how='inner')
v2 = v2.rename(columns={'vihecle_type_final':'vihecle_type'})

# 充电时间
v2['start_time'] = pd.to_datetime(v2['start_time'])
v2['start_hour'] = v2['start_time'].dt.hour
v2_1 = v2[v2['vihecle_type']==1].reset_index(drop=True)
v2_0 = v2[v2['vihecle_type']==0].reset_index(drop=True)

v2_1_count = v2_1[['start_hour', 'charging_id']].groupby('start_hour').count()
v2_0_count = v2_0[['start_hour', 'charging_id']].groupby('start_hour').count()

width = 0.8
x = np.arange(24)
plt.bar(x+width/4, v2_0_count['charging_id'], width/2, label='taxi')
plt.bar(x-width/4, v2_1_count['charging_id'], width/2, label='private_car')
plt.xticks(np.arange(24))
plt.title('charging count every hour')
plt.xlabel('hour')
plt.ylabel('count')
plt.legend()
plt.show()

# 充电时长
v2['charging_duration'] = v2['charging_duration'].astype(int)/60
v2['charging_duration'].hist(bins=100)

v2_1 = v2[v2['vihecle_type']==1].reset_index(drop=True)
v2_0 = v2[v2['vihecle_type']==0].reset_index(drop=True)

plt.hist(v2_1['charging_duration'], bins=100, alpha=0.6, label='private_car')
plt.hist(v2_0['charging_duration'], bins=100, alpha=0.6, label='taxi')
plt.xticks(np.arange(11)*2)
plt.xlabel('charging duration')
plt.ylabel('count')
plt.legend()
plt.show()



df_charging_type['vihecle_type'] = df_charging_type['vihecle_type'].astype(int)
df1['delta_soc'] = df1['delta_soc'].astype(int)
df1['charging_efficiency'] = df1['delta_soc']/(df1['charging_duration']/60)
df1 = pd.merge(df1, df_charging_type, how='left', on='deviceid')
plt.hist(df1[df1['vihecle_type']==1]['charging_efficiency'], alpha=0.5, bins=50, label='private_car')
plt.hist( df1[df1['vihecle_type']==0]['charging_efficiency'], alpha=0.5, bins=50, label='taxi')
plt.legend()
plt.show()






# 上海的经纬度是东经120°52′-122°12′,北纬30°40′-31°53′之间
# 上海经纬度：东经120.86-122.2，北纬30.67-31.88
df_sample['lg'] = pd.to_numeric(df_sample['lg'])
df_sample['lat'] = pd.to_numeric(df_sample['lat'])
df_sample['trip_distance'] = pd.to_numeric(df_sample['trip_distance'])
df_sh = df_sample[(df_sample['lg']>120.86) & (df_sample['lg']<122.2) & (df_sample['lat']>30.67) & (df_sample['lat']<31.88)]
df_err = df_sample[df_sample['lg']<10][['deviceid', 'trip_id']]
df_err = df_err.drop_duplicates()
df_err.loc[:, 'key'] = df_err['deviceid'] + df_err['trip_id'].astype(str)
df_sh.loc[:, 'key'] = df_sh['deviceid'] + df_sh['trip_id'].astype(str)
df_sh = df_sh[df_sh['key'].isin(df_err['key'].tolist()).map({True:False, False:True})]


df_vins = df_sh[['deviceid', 'trip_id', 'trip_distance']].drop_duplicates()
df_vins = df_vins[(df_vins['trip_distance']>8) & (df_vins['trip_distance']<12)]

vins = df_vins['deviceid'].unique().tolist()
df = df_vins[df_vins['deviceid']==vins[0]].iloc[:1, :]
for vin in vins[1:]:
    df = pd.concat([df, df_vins[df_vins['deviceid']==vin].iloc[:2, :]], ignore_index=True)

df_plot = df_sample[['deviceid', 'trip_id', 'lg', 'lat']]
df = pd.merge(df_plot, df, how='inner', on=['deviceid', 'trip_id'])

def update_points(num):
        if num%2==0:
            point_ani_1.set_marker("*")
            point_ani_1.set_color('g')
            point_ani_1.set_markersize(10)
            
            point_ani_2.set_marker("*")
            point_ani_2.set_color('g')
            point_ani_2.set_markersize(10)
            
            point_ani_3.set_marker("*")
            point_ani_3.set_color('g')
            point_ani_3.set_markersize(10)
            
            point_ani_4.set_marker("*")
            point_ani_4.set_color('g')
            point_ani_4.set_markersize(10)
            
            point_ani_5.set_marker("*")
            point_ani_5.set_color('g')
            point_ani_5.set_markersize(10)
            
            point_ani_6.set_marker("*")
            point_ani_6.set_color('g')
            point_ani_6.set_markersize(10)
            
            point_ani_7.set_marker("*")
            point_ani_7.set_color('g')
            point_ani_7.set_markersize(10)
            
            point_ani_8.set_marker("*")
            point_ani_8.set_color('g')
            point_ani_8.set_markersize(10)
            
            point_ani_9.set_marker("*")
            point_ani_9.set_color('g')
            point_ani_9.set_markersize(10)
            
            point_ani_10.set_marker("*")
            point_ani_10.set_color('g')
            point_ani_10.set_markersize(10)
            
        else:
            point_ani_1.set_marker("o")
            point_ani_1.set_color('r')
            point_ani_1.set_markersize(10)
            #point_ani.set_alpha=0.01      #不起作用
            
            point_ani_2.set_marker("o")
            point_ani_2.set_color('r')
            point_ani_2.set_markersize(10)
            
            point_ani_3.set_marker("o")
            point_ani_3.set_color('r')
            point_ani_3.set_markersize(10)
            
            point_ani_4.set_marker("o")
            point_ani_4.set_color('r')
            point_ani_4.set_markersize(10)
            
            point_ani_5.set_marker("o")
            point_ani_5.set_color('r')
            point_ani_5.set_markersize(10)
            
            point_ani_6.set_marker("o")
            point_ani_6.set_color('r')
            point_ani_6.set_markersize(10)
            
            point_ani_7.set_marker("o")
            point_ani_7.set_color('r')
            point_ani_7.set_markersize(10)
            
            point_ani_8.set_marker("o")
            point_ani_8.set_color('r')
            point_ani_8.set_markersize(10)
            
            point_ani_9.set_marker("o")
            point_ani_9.set_color('r')
            point_ani_9.set_markersize(10)
            
            point_ani_10.set_marker("o")
            point_ani_10.set_color('r')
            point_ani_10.set_markersize(10)
    
    
        point_ani_1.set_data(x0[num], y0[num])
        point_ani_2.set_data(x1[num], y1[num])
        point_ani_3.set_data(x2[num], y2[num])
        point_ani_4.set_data(x3[num], y3[num])
        point_ani_5.set_data(x4[num], y4[num])
        point_ani_6.set_data(x5[num], y5[num])
        point_ani_7.set_data(x6[num], y6[num])
        point_ani_8.set_data(x7[num], y7[num])
        point_ani_9.set_data(x8[num], y8[num])
        point_ani_10.set_data(x9[num], y9[num])
    #    text_pt.set_position((x[num], y[num]))
    #    text_pt.set_text("x=%.3f, y=%.3f" % (x[num], y[num]))
        
        return point_ani_1

data_len = 70
fig, ax  = plt.subplots()
x0 = df[df['deviceid']==vins[0]]['lg'].reset_index(drop=True)[:data_len]
y0 = df[df['deviceid']==vins[0]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x0, y0)
point_ani_1, = ax.plot(x0[0], y0[0], 'ro')

x1 = df[df['deviceid']==vins[1]]['lg'].reset_index(drop=True)[:data_len]
y1 = df[df['deviceid']==vins[1]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x1, y1)
point_ani_2, = ax.plot(x1[0], y1[0], 'ro')

x2 = df[df['deviceid']==vins[2]]['lg'].reset_index(drop=True)[:data_len]
y2 = df[df['deviceid']==vins[2]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x2, y2)
point_ani_3, = ax.plot(x2[0], y2[0], 'ro')

x3 = df[df['deviceid']==vins[3]]['lg'].reset_index(drop=True)[:data_len]
y3 = df[df['deviceid']==vins[3]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x3, y3)
point_ani_4, = ax.plot(x3[0], y3[0], 'ro')

x4 = df[df['deviceid']==vins[4]]['lg'].reset_index(drop=True)[:data_len]
y4 = df[df['deviceid']==vins[4]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x4, y4)
point_ani_5, = ax.plot(x4[0], y4[0], 'ro')

x5 = df[df['deviceid']==vins[5]]['lg'].reset_index(drop=True)[:data_len]
y5 = df[df['deviceid']==vins[5]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x5, y5)
point_ani_6, = ax.plot(x5[0], y5[0], 'ro')

x6 = df[df['deviceid']==vins[6]]['lg'].reset_index(drop=True)[:data_len]
y6 = df[df['deviceid']==vins[6]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x6, y6)
point_ani_7, = ax.plot(x6[0], y6[0], 'ro')

x7 = df[df['deviceid']==vins[7]]['lg'].reset_index(drop=True)[:data_len]
y7 = df[df['deviceid']==vins[7]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x7, y7)
point_ani_8, = ax.plot(x7[0], y7[0], 'ro')

x8 = df[df['deviceid']==vins[8]]['lg'].reset_index(drop=True)[:data_len]
y8 = df[df['deviceid']==vins[8]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x8, y8)
point_ani_9, = ax.plot(x8[0], y8[0], 'ro')

x9 = df[df['deviceid']==vins[9]]['lg'].reset_index(drop=True)[:data_len]
y9 = df[df['deviceid']==vins[9]]['lat'].reset_index(drop=True)[:data_len]
ax.plot(x9, y9)
point_ani_10, = ax.plot(x9[0], y9[0], 'ro')


ani = animation.FuncAnimation(fig=fig, func=update_points, frames=data_len, interval=200)

plt.grid(ls="--")
plt.show()

ani.save('D:/AI/machine_learning/matplotlib/vehicle.gif')


# 通勤特征
df_com =  pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_commuting_feature/dbh_commuting_feature.csv')
df_com['deviceid'].unique().shape
df_com.dtypes
#deviceid                    object
#trip_id                      int64
#start_time                  object
#stop_time                   object
#trip_duration              float64
#trip_distance              float64
#start_lg                   float64
#start_lat                  float64
#stop_lg                    float64
#stop_lat                   float64
#avg_speed                  float64
#stop_times                   int64
#delta_soc                    int64
#nasty_times                  int64
#home_lg                    float64
#home_lat                   float64
#company_lg                 float64
#company_lat                float64
#stop_company_distance      float64
#to_company_times             int64
#start_hour                   int64
#start_minute                 int64
#end_minute                   int64
#start_range_1               object
#start_range_2               object
#start_range                 object
#day_period                  object
#commuting_avg_duration     float64
#commuting_avg_distance     float64
#commuting_avg_speed        float64
#commuting_avg_delta_soc    float64

df1 = df_com[['deviceid','day_period','trip_duration','trip_distance','avg_speed','stop_times','delta_soc','nasty_times',
              'start_range','commuting_avg_duration','commuting_avg_distance','commuting_avg_speed','commuting_avg_delta_soc']]

# 通勤时间分布
height = 40
df1.trip_duration.describe()
df1.trip_duration.hist(bins=50)
plt.plot([24.7]*height , np.arange(height), 'r--')
plt.title('commuting duration distribution', fontsize=14)
plt.xlabel('commuting duration', fontsize=14)
plt.text(26, 32, 'μ=24.7, σ=18.6', fontsize=16)

# 通勤距离分布
height = 53
df1.trip_distance.describe()
df1.trip_distance.hist(bins=50)
plt.plot([10.1]*height , np.arange(height), 'r--')
plt.title('commuting distance distribution', fontsize=14)
plt.xlabel('commuting distance', fontsize=14)
plt.text(11, 47, 'μ=10.1, σ=8.9', fontsize=16)

# 通勤平均速度分布
df1.avg_speed.describe()
height = 30
μ = 33.9
σ = 8.9
df1.avg_speed.hist(bins=50)
plt.plot([μ]*height , np.arange(height), 'r--')
plt.title('commuting avg_speed distribution', fontsize=14)
plt.xlabel('commuting avg_speed', fontsize=14)
plt.text(μ+1, 27, 'μ={}, σ={}'.format(μ, σ), fontsize=16)

# 通勤耗电量分布
df1.delta_soc.describe()
height = 70
μ = 4.0
σ = 3.9
df1.delta_soc.hist(bins=30)
plt.plot([μ]*height , np.arange(height), 'r--')
plt.title('commuting delta_soc distribution', fontsize=14)
plt.xlabel('commuting delta_soc', fontsize=14)
plt.text(μ+1, 63, 'μ={}, σ={}'.format(μ, σ), fontsize=16)


# 上午、下午、晚上通勤时间分布对比
alpha = 0.65
df1[df1['day_period']=='Morning']['trip_duration'].hist(bins=20, alpha=alpha, label='Morning')
df1[df1['day_period']=='Afternoon']['trip_duration'].hist(bins=20,  alpha=alpha, label='Afternoon')
df1[df1['day_period']=='Evening']['trip_duration'].hist(bins=20,  alpha=alpha, label='Evening')
plt.legend(fontsize=12)
plt.title('commuting duration comparison', fontsize=14)
plt.xlabel('commuting duration', fontsize=14)


# 上午、下午、晚上通勤平均速度分布对比
alpha = 0.65
df1[df1['day_period']=='Morning']['avg_speed'].hist(bins=20, alpha=alpha, label='Morning')
df1[df1['day_period']=='Afternoon']['avg_speed'].hist(bins=20,  alpha=alpha, label='Afternoon')
df1[df1['day_period']=='Evening']['avg_speed'].hist(bins=20,  alpha=alpha, label='Evening')
plt.legend(fontsize=12)
plt.title('commuting avg_speed comparison', fontsize=14)
plt.xlabel('commuting avg_speed', fontsize=14)


# 通勤时间建议
df_com =  pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_commuting_advice/dbh_commuting_advice.csv')
df_com = df_com.sort_values(['deviceid', 'day_period'])


df1 = df_com[df_com['deviceid']=='LSVAY60E0K2010513'].reset_index(drop=True)
df2 = df_com[df_com['deviceid']=='LSVAY60E9K2012079'].reset_index(drop=True)
df_com2 = pd.concat([df1, df2], axis=0, ignore_index=True)

df1 = df_com[df_com['deviceid']=='LSVAY60E7K2010458'].reset_index(drop=True)
df_com2 = pd.concat([df_com2, df1], axis=0, ignore_index=True)

df1 = df_com[df_com['deviceid']=='LSVAY60E8K2010470'].reset_index(drop=True)
df_com2 = pd.concat([df_com2, df1], axis=0, ignore_index=True)

df1 = df_com[df_com['deviceid']=='LSVAX60E2K2016770'].reset_index(drop=True)
df_com2 = pd.concat([df_com2, df1], axis=0, ignore_index=True)









