# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:04:56 2020

@author: David
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import os
from scipy.stats import norm


# 行车安全（35%）-报警信息，权重35%
def warning(df_x):
    warning_score = df_x['mxal_times'].sum() + df_x['celohwn_times'].sum() + df_x['lsocwn_times'].sum()
    warning_score = (1 - min(1, np.abs(warning_score)))
    
    return warning_score

# 行车安全（35%）-按400次计算急加速分值：急加速一次扣分0.25%，权重20%
def rush(df_x):
    rush_times_total = df_x['rush_times'].sum()
    rush_times_avg = df_x['rush_times'].mean()
    # 根据总急加速次数计算得分
    if rush_times_total >= 400:
        score_total = 0
    else:
        score_total = (1 - rush_times_total * 0.0025)
    
    # 根据平均急加速次数计算得分
    score_avg = np.power(0.718, rush_times_avg)
    
    return 0.5*score_total + 0.5*score_avg

# 行车安全（35%）-按250次计算急减速分值，急减速一次扣分0.4%，权重35%
def nasty(df_x):
    nasty_times_total = df_s['nasty_times'].sum()
    nasty_times_avg = df_s['nasty_times'].mean()
    # 根据总急减速次数计算得分
    if nasty_times_total >= 250:
        score_total = 0
    else:
        score_total = (1 - nasty_times_total * 0.004)
    
    # 根据平均急减速次数计算得分
    score_avg = np.power(0.627, nasty_times_avg)
    
    return 0.5*score_total + 0.5*score_avg

# 超速得分：共20次机会，超速一次扣5%，权重10%
def overspeed(df_x):
    overspeed_times = df_s['speeding_times'].sum()
    if overspeed_times>=5:
        overspeed_score = 0
    else:
        overspeed_score = (1 - overspeed_times * 0.02)
    
    return overspeed_score

# 能耗表现（30%）
def soc(df_x):
    mu = 36
    std = 16
    delta_soc = df_x['delta_soc_per100km'].mean()
    if delta_soc <= mu:
        soc_score = 1
    else:
        gauss = norm(loc=mu, scale=std)
        soc_score = 40 * gauss.pdf(delta_soc)
    
    return soc_score

# 出行习惯（15%）：出行SOC低于20%一次，扣10%
def habit(df_x):
    less_20_times = np.sum((df_x['start_soc']<20) * 1)
    if less_20_times>=10:
        habit_score = 0
    else:
        habit_score = 1 - 0.1*less_20_times
        
    return habit_score

# 充电管理（10%）-充电效率得分（40%）
def efficiency(df_x):
    mean_efficiency = df_x['charging_efficiency'].mean()   # 判断慢充还是快充
    if mean_efficiency<=20:
        std = 1.37
        mu = 16.81 + std
        if mean_efficiency >= mu:
            mean_efficiency = mu
        gauss = norm(loc=mu, scale=std)
        df_slow = df_x[df_x['charging_efficiency']<=20].reset_index(drop=True)
        mean_efficiency = df_slow['charging_efficiency'].mean()
        efficiency_score = 3.4 * gauss.pdf(mean_efficiency)
        
    else:
        std = 24.6
        mu = 79.3 + std
        if mean_efficiency >= mu:
            mean_efficiency = mu
        gauss = norm(loc=mu, scale=std)
        df_fast = df_x[df_x['charging_efficiency']>20].reset_index(drop=True)
        mean_efficiency = df_fast['charging_efficiency'].mean()
        efficiency_score = 60 * gauss.pdf(mean_efficiency)
        
    return efficiency_score

# 充电管理（10%）-充电时长得分（30%）
def duration(df_x):
    mean_efficiency = df_x['charging_efficiency'].mean()   # 判断慢充还是快充
    if mean_efficiency <= 20:
        df_slow = df_x[df_x['charging_efficiency']<=20].reset_index(drop=True)
        duration_score = min(1.0, df_slow['charging_duration'].mean()/360)
        
    else:
        df_fast = df_x[df_x['charging_efficiency']>20].reset_index(drop=True)
        duration_score = min(1.0, df_fast['charging_duration'].mean()/76)
        
    return duration_score

# 充电管理（10%）-充电积极性（30%）：10pm后充电占比

def positivity(df_x):
    for i in np.arange(df_x.shape[0]):
        start_hour = df_x.loc[i, 'start_time'].hour
        end_hour = df_x.loc[i, 'stop_time'].hour
        
        if start_hour<6:
            start_hour += 24
        if end_hour<6:
            end_hour += 24
        
        if end_hour<start_hour:
            end_hour += 24
        
        if start_hour<22:
            if end_hour<22:
                positivity_time = 0
            elif end_hour<30:
                positivity_time = df_x.loc[i, 'stop_time'] - df_x.loc[i, 'start_time'].replace(hour=22, minute=0, second=0)
                positivity_time = int(positivity_time.seconds/60)
            else:
                positivity_time = 480
        
        elif start_hour<30:
            if end_hour<30:
                positivity_time = df_x.loc[i, 'stop_time'] - df_x.loc[i, 'start_time']
                positivity_time = int(positivity_time.seconds/60)
            else:
                positivity_time = df_x.loc[i, 'stop_time'].replace(hour=6, minute=0, second=0) - df_x.loc[i, 'start_time']
                positivity_time = int(positivity_time.seconds/60)
        
        else:
            positivity_time = 0
        
    return positivity_time / (df_x['charging_duration'].sum() + 1e-4)


#positivity_time = []
#for vin in vin_list:
#    df_charging_s = df_charging[df_charging['deviceid']==vin].reset_index(drop=True)
#    if df_charging_s.shape[0]==0:
#        positivity_time.append(0)
#        continue
#    positivity_time.append(positivity(df_charging_s))
#
#df_positivity_time = pd.DataFrame({'deviceid':vin_list, 'positivity_time':positivity_time})
#df_positivity_time = pd.merge(df_driving_behavior, df_positivity_time, on='deviceid')

# 环保指数（10%）
def environmental(df_x):
    miles = df_x['trip_distance'].sum()
    
    return 0.1*np.log(miles+1)



if __name__ == '__main__':
    # 数据读取
#    df_basic = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_trip_basic/dbh_trip_basic.csv')
    df_driving_behavior = pd.read_csv('D:/zhongdiao/上汽大众驾驶行为分析/data/pyspark_data/lavida_12/dbh_driving_behavior_12/dbh_driving_behavior.csv')
    df_charging = pd.read_csv('D:/zhongdiao/上汽大众驾驶行为分析/data/pyspark_data/lavida_12/dbh_charging_12/dbh_charging.csv')
    df_charging = df_charging[['deviceid', 'start_time', 'stop_time', 'charging_duration', 'delta_soc']]
        
    # 计算超速次数（使用行程基础数据）
#    df_basic['speed_greater_120'] = (df_basic['vehiclespeed']>120) * 1
#    df = df_basic[['deviceid', 'trip_id', 'speed_greater_120']]
#    df_count = df.groupby(['deviceid', 'trip_id'], as_index=False).sum()
    
    # 驾驶行为数据匹配超速次数
#    df = pd.merge(df_driving_behavior, df_count, how='inner', on=['deviceid', 'trip_id'])
    
    # 计算充电效率
    df_charging['charging_efficiency'] = df_charging['delta_soc']/(df_charging['charging_duration']+0.1) * 60
    df_charging = df_charging[df_charging['charging_efficiency']<=300].reset_index(drop=True)
    
    df_charging['start_time'] = pd.to_datetime(df_charging['start_time'])
    df_charging['stop_time'] = pd.to_datetime(df_charging['stop_time'])


    # vin_list
    vin_list = df_driving_behavior['deviceid'].unique().tolist()
    soc_score_list = []
    score_list = []
    i = 0
    for vin in vin_list:
#        vin = 'LSVAY60E5K2011348'
        df_s = df_driving_behavior[df_driving_behavior['deviceid']==vin].reset_index(drop=True)
        
        # 行车安全-报警信息、急加速、急减速、超速
        warning_score = warning(df_s)
        rush_score = rush(df_s)
        nasty_score = nasty(df_s)
        overspeed_score = overspeed(df_s)
        # 行车安全得分（35%）
        safe_score = 0.35 * warning_score + 0.35 * nasty_score + 0.2 * rush_score + 0.1 * overspeed_score
        
        # 能耗表现（30%）
        soc_score = soc(df_s)
        soc_score_list.append(soc_score)
        
        # 出行习惯（15%）
        habit_score = habit(df_s)
        
        # 充电管理（10%）
        # 获取样本
        df_charging_s = df_charging[df_charging['deviceid']==vin].reset_index(drop=True)
        if df_charging_s.shape[0]==0:
            print('{} no charging records'.format(vin))
            continue
        # 充电效率
        efficiency_score = efficiency(df_charging_s)
        # 充电时长
        duration_score = duration(df_charging_s)
        # 充电积极性
        positivity_score = positivity(df_charging_s)
#        print('positivity_score:', positivity_score)
        # 充电管理得分
        charging_score = 0.4*efficiency_score + 0.3*duration_score + 0.3*positivity_score
        
        # 环保指数（10%）
        environmental_score = environmental(df_s)
    
        # 总得分
        score =  int(500 * (0.35*safe_score + 0.3*soc_score + 0.15*habit_score + 0.1*charging_score + 0.1*environmental_score))
        print('step:{}, vin:{}, score:{}'.format(i, vin, score))
        i += 1
        score_list.append(score)

    plt.hist(score_list, bins=30)
    plt.title('2020-12 score distribution')
    plt.xlabel('score')






